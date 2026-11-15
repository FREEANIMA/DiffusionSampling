(() => {
  const $ = (id) => document.getElementById(id);
  const modeEl = $('mode');

  // ---------- 자동완성 ----------
  const TAGS = window.TAG_LIST || [];
  // 비교용 키: 소문자 + 공백/언더스코어 통일
  const key = (s) => s.trim().toLowerCase().replace(/[\s_]+/g, ' ');
  const KEYED = TAGS.map((t) => [t, key(t)]);

  const acEl = $('ac');
  const tagsEl = $('tags');
  let acItems = [], acIdx = -1;

  const lev = (a, b) => {
    const m = a.length, n = b.length;
    if (Math.abs(m - n) > 2) return 99;
    let prev = Array.from({ length: n + 1 }, (_, j) => j);
    for (let i = 1; i <= m; i++) {
      const cur = [i];
      for (let j = 1; j <= n; j++) {
        cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1,
                          prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      }
      prev = cur;
    }
    return prev[n];
  };

  const currentToken = () => {
    const v = tagsEl.value;
    const pos = tagsEl.selectionStart ?? v.length;
    const start = v.lastIndexOf(',', pos - 1) + 1;
    return { start, end: pos, text: v.slice(start, pos) };
  };

  const escapeHtml = (s) =>
    String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  const search = (raw) => {
    const q = key(raw);
    if (!q) return [];
    const pre = [], sub = [];
    for (const [t, k] of KEYED) {
      if (k.startsWith(q)) pre.push(t);
      else if (k.includes(q)) sub.push(t);
      if (pre.length >= 8) break;
    }
    let out = pre.concat(sub).slice(0, 8);
    if (out.length === 0 && q.length >= 3) {      // 후보 없으면 오타 교정
      out = KEYED.map(([t, k]) => [t, lev(q, k)])
                 .filter(([, d]) => d <= 2)
                 .sort((a, b) => a[1] - b[1])
                 .slice(0, 8).map(([t]) => t);
    }
    return out;
  };

  const closeAc = () => { acEl.style.display = 'none'; acIdx = -1; };
  const renderAc = () => {
    if (!acItems.length) return closeAc();
    acEl.innerHTML = acItems.map((t, i) =>
      `<div class="ac-item${i === acIdx ? ' on' : ''}" data-i="${i}">${escapeHtml(t)}</div>`
    ).join('');
    acEl.style.display = 'block';
  };

  const pick = (tag) => {
    const { start, end } = currentToken();
    const v = tagsEl.value;
    const before = v.slice(0, start).replace(/\s+$/, '');
    const sep = before ? (before.endsWith(',') ? ' ' : ', ') : '';
    const after = v.slice(end).replace(/^\s*,?\s*/, '');
    tagsEl.value = before + sep + tag + ', ' + after;
    closeAc();
    tagsEl.focus();
  };

  tagsEl.addEventListener('input', () => {
    acItems = search(currentToken().text);
    acIdx = -1;
    renderAc();
  });
  tagsEl.addEventListener('keydown', (e) => {
    if (acEl.style.display === 'none') return;
    if (e.key === 'ArrowDown') { acIdx = (acIdx + 1) % acItems.length; renderAc(); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { acIdx = (acIdx - 1 + acItems.length) % acItems.length; renderAc(); e.preventDefault(); }
    else if ((e.key === 'Enter' || e.key === 'Tab') && acIdx >= 0) { pick(acItems[acIdx]); e.preventDefault(); }
    else if (e.key === 'Escape') closeAc();
  });
  acEl.addEventListener('mousedown', (e) => {
    const i = e.target.dataset.i;
    if (i !== undefined) { pick(acItems[+i]); e.preventDefault(); }
  });
  tagsEl.addEventListener('blur', () => setTimeout(closeAc, 100));

  // ---------- 기존 UI ----------
  modeEl.onchange = () => {
    const isT2I = modeEl.value === 'txt2img';
    $('txt2img-box').style.display = isT2I ? 'block' : 'none';
    $('inpaint-box').style.display = isT2I ? 'none' : 'block';
  };

  $('clear_out').onclick = () => { $('out').textContent = ''; };

  $('run').onclick = () => {
    window.ui_params = {
      mode: modeEl.value,
      tags: $('tags').value,
      nsfw: $('nsfw').value === 'true',
      sampler: $('sampler').value,
      cfg: $('cfg').value,
      steps: $('steps').value,
      seed: $('seed').value,
      H: $('H').value,
      W: $('W').value,
      img_path: $('img_path').value,
      start_t: $('start_t').value,
    };
    $('out').textContent = 'running...';
  };
})();