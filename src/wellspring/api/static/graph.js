import { toast } from './helpers.js';

let graphData = null;
let sim = null;
let zoomBehavior = null;
let svg = null;
let g = null;
let ctxNode = null;
let getConf = () => 0;

/* entity type → color */
const TYPE_COLORS = {
  malware: '#ef4444',
  threat_actor: '#f97316',
  attack_pattern: '#a855f7',
  tool: '#3b82f6',
  vulnerability: '#eab308',
  campaign: '#ec4899',
  indicator: '#14b8a6',
  infrastructure: '#6366f1',
  mitigation: '#22c55e',
  report: '#64748b',
  identity: '#0ea5e9',
};
function nodeColor(d) { return TYPE_COLORS[d.type] || '#9ca3af'; }

export function initGraph(getConfidence) {
  getConf = getConfidence;

  document.getElementById('zoomInBtn').addEventListener('click', () => {
    if (svg) svg.transition().duration(300).call(zoomBehavior.scaleBy, 1.4);
  });
  document.getElementById('zoomOutBtn').addEventListener('click', () => {
    if (svg) svg.transition().duration(300).call(zoomBehavior.scaleBy, 0.7);
  });
  document.getElementById('fitBtn').addEventListener('click', fitToView);
  document.getElementById('pinBtn').addEventListener('click', togglePinAll);
  document.getElementById('clearGraphBtn').addEventListener('click', clearGraph);
  document.getElementById('exportStixBtn').addEventListener('click', exportStix);

  // context menu actions
  document.getElementById('ctxExpand').addEventListener('click', () => {
    if (ctxNode) expandNode(ctxNode);
    hideCtxMenu();
  });
  document.getElementById('ctxPin').addEventListener('click', () => {
    if (ctxNode) togglePin(ctxNode);
    hideCtxMenu();
  });
  document.getElementById('ctxExplain').addEventListener('click', async () => {
    if (!ctxNode) return;
    hideCtxMenu();
    try {
      const res = await fetch('/explain?entity_id=' + encodeURIComponent(ctxNode.id));
      const data = await res.json();
      const n = data.relations?.length || 0;
      toast(`${ctxNode.name}: ${n} relation(s) with provenance`, 'success');
    } catch (e) {
      toast('Could not load provenance', 'error');
    }
  });
  document.getElementById('ctxRemove').addEventListener('click', () => {
    if (ctxNode) removeNode(ctxNode.id);
    hideCtxMenu();
  });

  // STIX export handler
  async function exportStix() {
    if (!graphData || !graphData.nodes.length) {
      toast('Load a graph first', 'error');
      return;
    }
    // Use the first node as seed
    const seed = graphData.nodes[0];
    try {
      const res = await fetch(`/api/export/stix?seed_id=${encodeURIComponent(seed.id)}&depth=2`);
      if (!res.ok) throw new Error('Export failed');
      const bundle = await res.json();
      const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `wellspring-stix-${new Date().toISOString().slice(0,10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
      toast(`Exported ${bundle.objects?.length || 0} STIX objects`, 'success');
    } catch (e) {
      toast(e.message, 'error');
    }
  }

  window.addEventListener('resize', () => {
    if (svg) {
      const area = document.getElementById('graphArea');
      svg.attr('width', area.clientWidth).attr('height', area.clientHeight);
    }
  });
}

/* ── public: load query and render ─────── */
export async function loadGraph(seedName, depth, minConf) {
  const btn = document.getElementById('vizBtn');
  btn.disabled = true;
  btn.textContent = 'Loading...';
  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ seed_name: seedName, depth, min_confidence: minConf }),
    });
    if (!res.ok) {
      const e = await res.json();
      throw new Error(e.detail || 'Query failed');
    }
    const data = await res.json();
    renderGraph(data);
    toast(`Loaded ${data.nodes.length} nodes, ${data.edges.length} edges`, 'success');
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Visualize';
  }
}

/* ── clear ─────────────────────────────── */
function clearGraph() {
  const area = document.getElementById('graphArea');
  const oldSvg = area.querySelector('svg');
  if (oldSvg) oldSvg.remove();
  document.getElementById('graphEmpty').style.display = 'flex';
  document.getElementById('graphToolbar').style.display = 'none';
  if (sim) sim.stop();
  sim = null;
  graphData = null;
}

/* ── render ────────────────────────────── */
function renderGraph(data) {
  clearGraph();
  if (!data.nodes.length) { toast('No data to display', 'error'); return; }
  graphData = data;

  document.getElementById('graphEmpty').style.display = 'none';
  document.getElementById('graphToolbar').style.display = 'flex';

  const area = document.getElementById('graphArea');
  const W = area.clientWidth;
  const H = area.clientHeight;

  zoomBehavior = d3.zoom().scaleExtent([0.1, 8]).on('zoom', e => g.attr('transform', e.transform));

  svg = d3.select('#graphArea')
    .append('svg')
    .attr('width', W)
    .attr('height', H)
    .call(zoomBehavior);

  svg.on('click', () => hideCtxMenu());

  g = svg.append('g');

  // build links array for d3.forceLink
  const links = data.edges.map(e => ({
    id: e.id,
    source: e.subject_id,
    target: e.object_id,
    predicate: e.predicate,
    confidence: e.confidence,
    origin: e.attrs?.origin || 'extracted',
  }));

  const link = g.selectAll('.link')
    .data(links)
    .enter().append('line')
    .attr('class', d => {
      if (d.origin === 'inferred') return 'link inferred';
      if (d.origin === 'cooccurrence') return 'link cooccurrence';
      return 'link';
    })
    .attr('stroke-width', d => 1 + d.confidence);

  const edgeLabel = g.selectAll('.edge-label')
    .data(links)
    .enter().append('text')
    .attr('class', 'edge-label')
    .text(d => d.predicate);

  const node = g.selectAll('.node')
    .data(data.nodes)
    .enter().append('g')
    .attr('class', 'node')
    .call(d3.drag()
      .on('start', dragstarted)
      .on('drag', dragged)
      .on('end', dragended));

  node.append('circle')
    .attr('r', d => {
      const deg = links.filter(l =>
        l.source === d.id || l.target === d.id ||
        l.source.id === d.id || l.target.id === d.id
      ).length;
      return 10 + Math.min(deg * 2, 12);
    })
    .attr('fill', d => nodeColor(d))
    .attr('stroke', '#1e293b')
    .attr('stroke-width', 1.5)
    .style('cursor', 'pointer')
    .on('mouseover', function() { d3.select(this).attr('stroke-width', 2.5); })
    .on('mouseout', function() { d3.select(this).attr('stroke-width', 1.5); })
    .on('contextmenu', (event, d) => { event.preventDefault(); showCtxMenu(event, d); })
    .on('dblclick', (event, d) => { event.stopPropagation(); expandNode(d); });

  node.append('text')
    .attr('class', 'node-label')
    .attr('x', 18)
    .attr('y', 4)
    .text(d => d.name);

  sim = d3.forceSimulation(data.nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(160))
    .force('charge', d3.forceManyBody().strength(-400))
    .force('center', d3.forceCenter(W / 2, H / 2))
    .force('collision', d3.forceCollide(30));

  sim.on('tick', () => {
    link
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);

    node.attr('transform', d => `translate(${d.x},${d.y})`);

    edgeLabel
      .attr('x', d => (d.source.x + d.target.x) / 2)
      .attr('y', d => (d.source.y + d.target.y) / 2);
  });
}

/* ── drag handlers ─────────────────────── */
function dragstarted(event) {
  if (!event.active) sim.alphaTarget(0.3).restart();
  event.subject.fx = event.subject.x;
  event.subject.fy = event.subject.y;
}
function dragged(event) {
  event.subject.fx = event.x;
  event.subject.fy = event.y;
}
function dragended(event) {
  if (!event.active) sim.alphaTarget(0);
  // nodes stay pinned after drag
}

/* ── context menu ──────────────────────── */
function showCtxMenu(event, d) {
  ctxNode = d;
  const menu = document.getElementById('ctxMenu');
  const rect = document.getElementById('graphArea').getBoundingClientRect();
  menu.style.left = (event.clientX - rect.left) + 'px';
  menu.style.top = (event.clientY - rect.top) + 'px';
  menu.classList.add('show');
  document.getElementById('ctxPin').textContent = d.fx != null ? 'Unpin node' : 'Pin node';
}

function hideCtxMenu() {
  document.getElementById('ctxMenu').classList.remove('show');
  ctxNode = null;
}

/* ── node operations ───────────────────── */
function togglePin(d) {
  if (d.fx != null) { d.fx = null; d.fy = null; }
  else { d.fx = d.x; d.fy = d.y; }
  sim.alpha(0.1).restart();
}

function removeNode(nodeId) {
  graphData.nodes = graphData.nodes.filter(n => n.id !== nodeId);
  graphData.edges = graphData.edges.filter(e => e.subject_id !== nodeId && e.object_id !== nodeId);
  renderGraph(graphData);
  toast('Node removed', 'success');
}

async function expandNode(d) {
  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ seed_id: d.id, depth: 1, min_confidence: getConf() }),
    });
    if (!res.ok) throw new Error('Expand failed');
    const data = await res.json();
    const existingIds = new Set(graphData.nodes.map(n => n.id));
    const existingEdges = new Set(graphData.edges.map(e => e.id));
    let added = 0;
    data.nodes.forEach(n => { if (!existingIds.has(n.id)) { graphData.nodes.push(n); added++; } });
    data.edges.forEach(e => { if (!existingEdges.has(e.id)) graphData.edges.push(e); });
    renderGraph(graphData);
    toast(`Expanded: +${added} nodes`, 'success');
  } catch (e) {
    toast(e.message, 'error');
  }
}

/* ── toolbar actions ───────────────────── */
function fitToView() {
  if (!graphData || !graphData.nodes.length) return;
  const area = document.getElementById('graphArea');
  const W = area.clientWidth, H = area.clientHeight;
  const xs = graphData.nodes.map(n => n.x || 0);
  const ys = graphData.nodes.map(n => n.y || 0);
  const x0 = Math.min(...xs) - 60, x1 = Math.max(...xs) + 60;
  const y0 = Math.min(...ys) - 60, y1 = Math.max(...ys) + 60;
  const scale = Math.min(W / (x1 - x0), H / (y1 - y0), 2);
  const tx = W / 2 - scale * (x0 + x1) / 2;
  const ty = H / 2 - scale * (y0 + y1) / 2;
  svg.transition().duration(500)
    .call(zoomBehavior.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
}

function togglePinAll() {
  if (!graphData) return;
  const anyPinned = graphData.nodes.some(n => n.fx != null);
  graphData.nodes.forEach(n => {
    if (anyPinned) { n.fx = null; n.fy = null; }
    else { n.fx = n.x; n.fy = n.y; }
  });
  sim.alpha(0.1).restart();
  toast(anyPinned ? 'All nodes unpinned' : 'All nodes pinned', 'success');
}
