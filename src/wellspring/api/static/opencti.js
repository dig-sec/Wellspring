import { toast } from './helpers.js';

export function initOpenCTI() {
  document.getElementById('scanFilesBtn').addEventListener('click', scanFiles);

  // Load watched folder names into button label
  loadWatchedFolders();

  // Poll stats every 10s
  refreshStats();
  setInterval(refreshStats, 10000);

  // Check for any running tasks on load
  checkTasks();
}

let watchedFolderLabel = 'Watched Folders';

async function loadWatchedFolders() {
  try {
    const res = await fetch('/api/watched-folders');
    const folders = await res.json();
    if (folders.length > 0) {
      const names = folders.map(f => f.path.split('/').pop()).join(', ');
      const total = folders.reduce((s, f) => s + f.file_count, 0);
      watchedFolderLabel = names;
      const btn = document.getElementById('scanFilesBtn');
      btn.innerHTML = `&#x1F4C1; Scan ${names} (${total.toLocaleString()} files)`;
    }
  } catch (e) { /* silent */ }
}

async function refreshStats() {
  try {
    const res = await fetch('/api/stats');
    const s = await res.json();
    const bar = document.getElementById('statsBar');
    bar.textContent = `${s.entities} entities · ${s.relations} rels · ${s.runs_pending} pending · ${s.runs_running} running · ${s.runs_completed} done`;
  } catch (e) { /* silent */ }
}

async function scanFiles() {
  const btn = document.getElementById('scanFilesBtn');
  btn.disabled = true;
  btn.textContent = 'Scanning...';

  try {
    // Scan all watched folders
    const res = await fetch('/api/scan', { method: 'POST' });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Scan failed');
    }

    const data = await res.json();
    toast(`Scan started (task ${data.task_id})`, 'success');
    pollTask(data.task_id);

  } catch (e) {
    toast(e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `&#x1F4C1; Scan ${watchedFolderLabel}`;
  }
}

function pollTask(taskId) {
  const bar = document.getElementById('taskBar');
  const badge = document.getElementById('taskBadge');
  const progress = document.getElementById('taskProgress');
  bar.style.display = 'flex';

  const iv = setInterval(async () => {
    try {
      const res = await fetch('/api/tasks/' + taskId);
      const t = await res.json();

      badge.textContent = t.status;
      badge.className = 'task-badge ' + t.status;
      progress.textContent = t.progress || '';

      if (t.status === 'completed' || t.status === 'failed') {
        clearInterval(iv);
        toast(
          t.status === 'completed' ? t.progress : `Task failed: ${t.error}`,
          t.status === 'completed' ? 'success' : 'error'
        );
        refreshStats();
        // Auto-hide bar after 15s
        setTimeout(() => { bar.style.display = 'none'; }, 15000);
      }
    } catch (e) {
      clearInterval(iv);
    }
  }, 2000);
}

async function checkTasks() {
  try {
    const res = await fetch('/api/tasks');
    const tasks = await res.json();
    const running = tasks.find(t => t.status === 'running');
    if (running) pollTask(running.id);
  } catch (e) { /* silent */ }
}
