import { initTabs, initSearch } from './sidebar.js';
import { initUpload, initRuns } from './ingest.js';
import { initGraph, loadGraph } from './graph.js';
import { initOpenCTI } from './opencti.js';

/* ── bootstrap ────────────────────────── */
initTabs();

const search = initSearch(seed => {
  loadGraph(seed, search.getDepth(), search.getConfidence());
});

initGraph(search.getConfidence);
initUpload();
initRuns();
initOpenCTI();
