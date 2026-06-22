/* ── Inline sequence viewer (window.OriViewer) ─────────────────────────────
   GenBank parse, feature colours, DNA text view + per-feature highlight,
   ORF/translation track, feature/DNA find, hover, right-click copy, expand.  */
(function () {
  'use strict';

  var COMPLEMENT = {
    A:'T', T:'A', G:'C', C:'G', U:'A', N:'N',
    R:'Y', Y:'R', S:'S', W:'W', K:'M', M:'K',
    B:'V', V:'B', D:'H', H:'D'
  };
  function comp(ch) { return COMPLEMENT[ch] || 'N'; }
  function revComp(s) {
    var o = '';
    for (var i = s.length - 1; i >= 0; i--) { o += comp(s[i]); }
    return o;
  }

  // 31-type feature colour map (Ori FEATURE_COLORS); unknown -> slate.
  var FEATURE_COLORS = {
    CDS:'#8b7fd6', gene:'#6ea8d6', promoter:'#d68b7f', terminator:'#c97fb0',
    rep_origin:'#7fd6a8', oriT:'#7fd6c4', RBS:'#f5c842', regulatory:'#e0a458',
    polyA_signal:'#d99bba', enhancer:'#e8915a', primer_bind:'#9bd07a',
    protein_bind:'#7fb3d6', misc_feature:'#94a3b8', misc_recomb:'#b08fd6',
    LTR:'#d6a87f', sig_peptide:'#a3d67f', mat_peptide:'#7fd690',
    intron:'#c0c0c0', exon:'#86b3e0', mRNA:'#86c5e0', "5'UTR":'#cdb88f',
    "3'UTR":'#cdb88f', tRNA:'#7fcdd6', rRNA:'#7fb8d6', ncRNA:'#a37fd6',
    mobile_element:'#d6b67f', stem_loop:'#d67f9b', repeat_region:'#bdb38f',
    source:'#cfd8dc', misc_RNA:'#9ec5d6', tag:'#e0b34f'
  };
  function featureColor(f) {
    var q = f.qualifiers || {};
    var c = (q.ApEinfo_fwdcolor && q.ApEinfo_fwdcolor[0]) ||
            (q.color && q.color[0]);
    if (c && /^#[0-9a-fA-F]{6}$/.test(c)) { return c.toLowerCase(); }
    return FEATURE_COLORS[f.type] || '#94a3b8';
  }
  function hexToRgba(hex, a) {
    var m = hex.replace('#', '');
    var r = parseInt(m.slice(0, 2), 16),
        g = parseInt(m.slice(2, 4), 16),
        b = parseInt(m.slice(4, 6), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
  }
  // pick a legible text colour for a solid feature-bar background
  function idealText(hex) {
    var m = hex.replace('#', '');
    var r = parseInt(m.slice(0, 2), 16),
        g = parseInt(m.slice(2, 4), 16),
        b = parseInt(m.slice(4, 6), 16);
    return (0.299 * r + 0.587 * g + 0.114 * b) > 150 ? '#23201c' : '#ffffff';
  }

  // Standard genetic code + per-amino-acid colours (for the ORF/translation track)
  var CODON_TABLE = {
    TTT:'F', TTC:'F', TTA:'L', TTG:'L', CTT:'L', CTC:'L', CTA:'L', CTG:'L',
    ATT:'I', ATC:'I', ATA:'I', ATG:'M', GTT:'V', GTC:'V', GTA:'V', GTG:'V',
    TCT:'S', TCC:'S', TCA:'S', TCG:'S', CCT:'P', CCC:'P', CCA:'P', CCG:'P',
    ACT:'T', ACC:'T', ACA:'T', ACG:'T', GCT:'A', GCC:'A', GCA:'A', GCG:'A',
    TAT:'Y', TAC:'Y', TAA:'*', TAG:'*', CAT:'H', CAC:'H', CAA:'Q', CAG:'Q',
    AAT:'N', AAC:'N', AAA:'K', AAG:'K', GAT:'D', GAC:'D', GAA:'E', GAG:'E',
    TGT:'C', TGC:'C', TGA:'*', TGG:'W', CGT:'R', CGC:'R', CGA:'R', CGG:'R',
    AGT:'S', AGC:'S', AGA:'R', AGG:'R', GGT:'G', GGC:'G', GGA:'G', GGG:'G'
  };
  function translateCodon(c) { return CODON_TABLE[c] || 'X'; }
  var AA_COLORS = {
    A:'#c6c6c6', G:'#c9b7dd', V:'#9ec06a', L:'#a8cf7c', I:'#8fbf6a',
    F:'#9a8ed6', M:'#aebf4e', P:'#d7a25f', S:'#6fb0e0', T:'#e3a44f',
    C:'#e3cf3e', N:'#73cdbd', Q:'#5fc6b4', Y:'#74bcd6', W:'#b193d4',
    D:'#e88aa0', E:'#e57f97', K:'#6f9fe0', R:'#5b8fdc', H:'#5fc4c4',
    '*':'#e25555', X:'#bfbfbf'
  };

  // ── GenBank location string -> {start,end,strand,spans} (1-based inclusive)
  function parseLocation(s) {
    s = (s || '').replace(/\s+/g, '');
    var strand = 1;
    if (/^complement\(/i.test(s)) {
      strand = -1;
      s = s.replace(/^complement\(/i, '');
      if (s.charAt(s.length - 1) === ')') { s = s.slice(0, -1); }
    }
    var jm = /^(?:join|order)\((.*)\)$/i.exec(s);
    var inner = jm ? jm[1] : s;
    var parts = inner.split(',');
    var spans = [];
    for (var i = 0; i < parts.length; i++) {
      var p = parts[i].replace(/^complement\(/i, '');
      if (p.charAt(p.length - 1) === ')') { p = p.slice(0, -1); }
      var mm = /(\d+)\.\.(\d+)/.exec(p);
      if (mm) {
        spans.push({ start: parseInt(mm[1], 10), end: parseInt(mm[2], 10) });
      } else {
        var single = /(\d+)/.exec(p);
        if (single) {
          var v = parseInt(single[1], 10);
          spans.push({ start: v, end: v });
        }
      }
    }
    if (!spans.length) { return { start: 1, end: 1, strand: strand, spans: [{ start: 1, end: 1 }] }; }
    var lo = spans[0].start, hi = spans[0].end;
    for (var k = 1; k < spans.length; k++) {
      if (spans[k].start < lo) { lo = spans[k].start; }
      if (spans[k].end > hi) { hi = spans[k].end; }
    }
    return { start: lo, end: hi, strand: strand, spans: spans };
  }

  // ── GenBank text -> {locus,size,topology,features,sequence}
  function parseGenBank(text) {
    var lines = String(text || '').split(/\r?\n/);
    var locus = '', size = 0, topology = 'linear';
    var features = [], seqParts = [];
    var section = 'header';
    var cur = null, curQual = null;

    function pushCur() {
      if (!cur) { return; }
      var L = parseLocation(cur.loc);
      features.push({
        type: cur.type, start: L.start, end: L.end,
        strand: L.strand, spans: L.spans, qualifiers: cur.qualifiers
      });
      cur = null; curQual = null;
    }

    for (var i = 0; i < lines.length; i++) {
      var ln = lines[i];
      if (ln.indexOf('//') === 0) { break; }

      if (section === 'header') {
        if (ln.indexOf('LOCUS') === 0) {
          var pp = ln.trim().split(/\s+/);
          if (pp[1]) { locus = pp[1]; }
          for (var j = 2; j < pp.length; j++) {
            if (/^bp$/i.test(pp[j]) && pp[j - 1]) { size = parseInt(pp[j - 1], 10) || 0; }
          }
          if (/\bcircular\b/i.test(ln)) { topology = 'circular'; }
        } else if (ln.indexOf('FEATURES') === 0) { section = 'features'; }
        else if (ln.indexOf('ORIGIN') === 0) { section = 'origin'; }
        continue;
      }

      if (section === 'features') {
        if (ln.indexOf('ORIGIN') === 0) { pushCur(); section = 'origin'; continue; }
        // New feature line: 5 leading spaces, then a non-space type token.
        if (ln.length > 5 && ln.slice(0, 5).replace(/ /g, '') === '' && ln.charAt(5) !== ' ') {
          pushCur();
          var rest = ln.slice(5);
          var sp = rest.indexOf(' ');
          var ftype, floc;
          if (sp < 0) { ftype = rest.trim(); floc = ''; }
          else { ftype = rest.slice(0, sp).trim(); floc = rest.slice(sp).trim(); }
          cur = { type: ftype || 'misc_feature', loc: floc, qualifiers: {} };
          curQual = null;
        } else if (cur) {
          var t = ln.trim();
          if (t.charAt(0) === '/') {
            var eq = t.indexOf('=');
            var key, val;
            if (eq < 0) { key = t.slice(1); val = ''; }
            else { key = t.slice(1, eq); val = t.slice(eq + 1); }
            var closed = true;
            if (val.charAt(0) === '"') {
              val = val.slice(1);
              if (val.charAt(val.length - 1) === '"') { val = val.slice(0, -1); }
              else { closed = false; }
            }
            if (!cur.qualifiers[key]) { cur.qualifiers[key] = []; }
            cur.qualifiers[key].push(val);
            curQual = closed ? null : key;
          } else if (curQual) {
            var arr = cur.qualifiers[curQual];
            var v = t, close = false;
            if (v.charAt(v.length - 1) === '"') { v = v.slice(0, -1); close = true; }
            arr[arr.length - 1] += (curQual === 'translation' ? '' : ' ') + v;
            if (close) { curQual = null; }
          } else {
            cur.loc += t; // continuation of a long location string
          }
        }
        continue;
      }

      if (section === 'origin') {
        var s = ln.replace(/[^A-Za-z]/g, '');
        if (s) { seqParts.push(s); }
      }
    }
    pushCur();

    var sequence = seqParts.join('').toUpperCase();
    if (!size) { size = sequence.length; }
    return { locus: locus, size: size, topology: topology, features: features, sequence: sequence };
  }

  function featureLabel(f) {
    var q = f.qualifiers || {};
    var keys = ['label', 'gene', 'product', 'standard_name', 'note'];
    for (var i = 0; i < keys.length; i++) {
      if (q[keys[i]] && q[keys[i]][0]) { return q[keys[i]][0]; }
    }
    return f.type;
  }

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }

  function toFasta(name, seq) {
    var out = '>' + name + '\n';
    for (var i = 0; i < seq.length; i += 70) { out += seq.slice(i, i + 70) + '\n'; }
    return out;
  }

  // Detect antibiotic-resistance markers from feature labels/qualifiers.
  var RESIST = [
    { re: /\b(?:ampr|bla|ampicillin|beta-?lactamase|carbenicillin)\b/i, ab: 'Ampicillin' },
    { re: /\b(?:kanr|neor|neo|kanamycin|neomycin|g418)\b/i, ab: 'Kanamycin/Neomycin' },
    { re: /\b(?:cmr|cat|chloramphenicol)\b/i, ab: 'Chloramphenicol' },
    { re: /\b(?:puror|pac|puromycin)\b/i, ab: 'Puromycin' },
    { re: /\b(?:hygr|hph|hygromycin)\b/i, ab: 'Hygromycin' },
    { re: /\b(?:blastr|bsd|blasticidin)\b/i, ab: 'Blasticidin' },
    { re: /\b(?:zeor|zeocin|bleomycin)\b/i, ab: 'Zeocin' },
    { re: /\b(?:specr|smr|aada|spectinomycin|streptomycin)\b/i, ab: 'Spectinomycin' },
    { re: /\b(?:gentr|gmr|gentamicin)\b/i, ab: 'Gentamicin' },
    { re: /\b(?:tetracycline|teta)\b/i, ab: 'Tetracycline' }
  ];
  function detectResistance(features) {
    var found = [], seen = {};
    for (var i = 0; i < features.length; i++) {
      var f = features[i], q = f.qualifiers || {};
      var hay = [featureLabel(f), f.type];
      ['gene', 'product', 'label', 'note'].forEach(function (k) { if (q[k]) { hay = hay.concat(q[k]); } });
      var s = hay.join(' ');
      for (var j = 0; j < RESIST.length; j++) {
        if (RESIST[j].re.test(s) && !seen[RESIST[j].ab]) { seen[RESIST[j].ab] = 1; found.push(RESIST[j].ab); }
      }
    }
    return found;
  }

  function downloadText(filename, text, mime) {
    var blob = new Blob([text], { type: mime || 'text/plain' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  var ICON_DL = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>';
  var ICON_PILL = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="8" width="18" height="8" rx="4"/><line x1="12" y1="8" x2="12" y2="16"/></svg>';

  // ── A single viewer instance bound to one panel element ──────────────────
  function Viewer(gb, name, rawGb) {
    this.gb = gb;
    this.name = name || gb.locus || 'construct';
    this.rawGb = rawGb || '';
    this.seq = gb.sequence;
    this.L = gb.sequence.length;
    this.showComp = false;
    this.showOrf = false;
    this.featList = gb.features.filter(function (f) { return f.type !== 'source'; });
    this.menu = null;
    this.expanded = false;
    this._backdrop = null;
    this._expandBtn = null;
    this.charMap = [];
    this.matches = [];
    this.activeIdx = -1;
    this.searchTimer = null;
    this.PER = 60;
    this.el = null;
    this.seqEl = null;
    this.countEl = null;
    this.prevBtn = null;
    this.nextBtn = null;
  }

  Viewer.prototype.gcPercent = function () {
    if (!this.L) { return 0; }
    var gc = 0;
    for (var i = 0; i < this.L; i++) {
      var c = this.seq[i];
      if (c === 'G' || c === 'C') { gc++; }
    }
    return Math.round((gc / this.L) * 1000) / 10;
  };

  // position(1-based) -> translucent colour (smallest covering feature wins) and
  // featAt[pos] -> list of features covering that base (for the hover tooltip)
  Viewer.prototype.buildColorMap = function () {
    var colorAt = new Array(this.L + 1);
    var featAt = new Array(this.L + 1);
    var L = this.L;
    function mark(p, f, rgba) {
      if (p < 1 || p > L) { return; }
      colorAt[p] = rgba;
      (featAt[p] = featAt[p] || []).push(f);
    }
    var feats = this.gb.features.slice().sort(function (a, b) {
      return (b.end - b.start) - (a.end - a.start); // largest first; smaller painted last wins
    });
    for (var i = 0; i < feats.length; i++) {
      var f = feats[i];
      if (f.type === 'source') { continue; }
      var rgba = hexToRgba(featureColor(f), 0.34);
      for (var s = 0; s < f.spans.length; s++) {
        var span = f.spans[s];
        var a = span.start, b = span.end;
        if (a <= b) {
          for (var p = a; p <= b; p++) { mark(p, f, rgba); }
        } else { // wraps the origin (circular)
          for (var p1 = a; p1 <= L; p1++) { mark(p1, f, rgba); }
          for (var p2 = 1; p2 <= b; p2++) { mark(p2, f, rgba); }
        }
      }
    }
    this.colorAt = colorAt;
    this.featAt = featAt;
  };

  // tick + number axis aligned under a row (monospace: 1 char == 1 base column)
  function buildRuler(rowStart, rowEnd) {
    var rlen = rowEnd - rowStart + 1;
    var ticks = new Array(rlen), nums = new Array(rlen);
    for (var t = 0; t < rlen; t++) { ticks[t] = ' '; nums[t] = ' '; }
    for (var pos = rowStart; pos <= rowEnd; pos++) {
      if (pos % 10 === 0) {
        var col = pos - rowStart;
        ticks[col] = '|';
        var ss = String(pos);
        for (var d = 0; d < ss.length; d++) { // right-align number's last digit under the tick
          var cc = col - (ss.length - 1) + d;
          if (cc >= 0 && cc < rlen) { nums[cc] = ss.charAt(d); }
        }
      }
    }
    return { ticks: ticks.join(''), nums: nums.join('') };
  }

  Viewer.prototype.renderSequence = function () {
    this.buildColorMap();
    var html = [];
    var seq = this.seq, L = this.L, PER = this.PER, colorAt = this.colorAt;
    var cw = this._charW || this._measureCharW();
    var trackFeats = this.featList;
    var LANE_H = 16, MAX_LANES = 6;
    for (var rowStart = 1; rowStart <= L; rowStart += PER) {
      var rowEnd = Math.min(rowStart + PER - 1, L);
      // Feature track in its OWN row above the DNA (Ori-style labelled bars), so
      // the left position number stays aligned to the DNA line, not the bars.
      var trackHtml = this.renderTrack(trackFeats, rowStart, rowEnd, cw, LANE_H, MAX_LANES);
      if (trackHtml) {
        html.push('<div class="oriv-row oriv-track-row"><span class="oriv-pos"></span>' +
          '<span class="oriv-stack">' + trackHtml + '</span></div>');
      }
      html.push('<div class="oriv-row"><span class="oriv-pos">' + rowStart + '</span><span class="oriv-stack">');
      // forward strand
      html.push('<span class="oriv-line oriv-fwd">');
      for (var p = rowStart; p <= rowEnd; p++) {
        var c = colorAt[p];
        html.push('<span class="oriv-base" data-pos="' + p + '"' +
          (c ? ' style="background:' + c + '"' : '') + '>' + seq[p - 1] + '</span>');
      }
      html.push('</span>');
      // complement strand (optional)
      if (this.showComp) {
        html.push('<span class="oriv-line oriv-comp">');
        for (var q = rowStart; q <= rowEnd; q++) {
          html.push('<span class="oriv-cbase">' + comp(seq[q - 1]) + '</span>');
        }
        html.push('</span>');
      }
      // amino-acid / ORF track (optional): one chevron per codon of each CDS
      if (this.showOrf) { html.push(this.renderAaTrack(rowStart, rowEnd, cw)); }
      // position ruler beneath the DNA (ticks + numbers every 10 bp), Ori-style
      var ruler = buildRuler(rowStart, rowEnd);
      html.push('<span class="oriv-line oriv-rule-ticks">' + esc(ruler.ticks) + '</span>');
      html.push('<span class="oriv-line oriv-rule-nums">' + esc(ruler.nums) + '</span>');
      html.push('</span></div>');
    }
    this.seqEl.innerHTML = html.join('');
    // index base spans by 1-based position for O(1) highlight
    this.charMap = new Array(L + 1);
    var nodes = this.seqEl.querySelectorAll('.oriv-base');
    for (var n = 0; n < nodes.length; n++) {
      this.charMap[parseInt(nodes[n].getAttribute('data-pos'), 10)] = nodes[n];
    }
  };

  // Feature track for one row: labelled colour bars, lane-packed so overlapping
  // features stack. Positioned in px from the measured char width so bars line
  // up exactly with the base columns. Returns '' if no feature touches the row.
  Viewer.prototype.renderTrack = function (feats, rowStart, rowEnd, cw, laneH, maxLanes) {
    var intervals = [];
    for (var fi = 0; fi < feats.length; fi++) {
      var f = feats[fi];
      for (var si = 0; si < f.spans.length; si++) {
        var a = f.spans[si].start, b = f.spans[si].end;
        if (a <= b) {
          var s1 = Math.max(a, rowStart), e1 = Math.min(b, rowEnd);
          if (s1 <= e1) { intervals.push({ f: f, fi: fi, s: s1, e: e1, contS: a < s1, contE: b > e1 }); }
        } else { // feature wraps the origin (circular)
          var s2 = Math.max(a, rowStart), e2 = Math.min(this.L, rowEnd);
          if (s2 <= e2) { intervals.push({ f: f, fi: fi, s: s2, e: e2, contS: a < s2, contE: true }); }
          var s3 = Math.max(1, rowStart), e3 = Math.min(b, rowEnd);
          if (s3 <= e3) { intervals.push({ f: f, fi: fi, s: s3, e: e3, contS: true, contE: b > e3 }); }
        }
      }
    }
    if (!intervals.length) { return ''; }
    intervals.sort(function (x, y) { return (x.s - y.s) || ((y.e - y.s) - (x.e - x.s)); });
    var laneEnds = [];
    for (var i = 0; i < intervals.length; i++) {
      var iv = intervals[i], placed = false;
      for (var k = 0; k < laneEnds.length; k++) {
        if (laneEnds[k] < iv.s) { iv.lane = k; laneEnds[k] = iv.e; placed = true; break; }
      }
      if (!placed) { iv.lane = laneEnds.length; laneEnds.push(iv.e); }
    }
    var nLanes = Math.min(laneEnds.length, maxLanes);
    var bars = [];
    for (var j = 0; j < intervals.length; j++) {
      var v = intervals[j];
      if (v.lane >= maxLanes) { continue; }
      var left = (v.s - rowStart) * cw;
      var width = (v.e - v.s + 1) * cw;
      var top = (nLanes - 1 - v.lane) * laneH;
      var col = featureColor(v.f);
      var label = (v.f.strand === -1 ? '◀ ' : (v.f.strand === 1 ? '▶ ' : '')) + featureLabel(v.f);
      var rng = (v.f.start === v.f.end) ? String(v.f.start) : (v.f.start + '–' + v.f.end);
      var rad = (v.contS ? '0' : '3px') + ' ' + (v.contE ? '0' : '3px') + ' ' +
                (v.contE ? '0' : '3px') + ' ' + (v.contS ? '0' : '3px');
      bars.push('<span class="oriv-feat-bar" data-fi="' + v.fi + '" title="' +
        esc(v.f.type + ' · ' + rng + (v.f.strand === -1 ? ' (reverse)' : '') + ' · right-click to copy') + '"' +
        ' style="left:' + left.toFixed(2) + 'px;width:' + width.toFixed(2) + 'px;top:' + top +
        'px;background:' + col + ';color:' + idealText(col) + ';border-radius:' + rad + '">' +
        esc(label) + '</span>');
    }
    return '<div class="oriv-track" style="height:' + (nLanes * laneH) + 'px">' + bars.join('') + '</div>';
  };

  // Translate every CDS feature into codons (cached). Each codon is one chevron
  // in the ORF track. Forward CDS read 5'->3'; reverse CDS read on the bottom
  // strand. (Multi-span/spliced CDS are translated per contiguous span.)
  Viewer.prototype.computeCodons = function () {
    if (this._codons) { return this._codons; }
    var codons = [], seq = this.seq, featList = this.featList;
    for (var fi = 0; fi < featList.length; fi++) {
      var f = featList[fi];
      if (f.type !== 'CDS') { continue; }
      for (var si = 0; si < f.spans.length; si++) {
        var a = f.spans[si].start, b = f.spans[si].end;
        if (a > b) { continue; } // origin-wrapped span: skip translation (rare)
        if (f.strand === -1) {
          for (var p = b; p - 2 >= a; p -= 3) {
            codons.push({ start: p - 2, end: p, aa: translateCodon(revComp(seq.substr(p - 3, 3))), strand: -1, fi: fi });
          }
        } else {
          for (var q = a; q + 2 <= b; q += 3) {
            codons.push({ start: q, end: q + 2, aa: translateCodon(seq.substr(q - 1, 3)), strand: 1, fi: fi });
          }
        }
      }
    }
    this._codons = codons;
    return codons;
  };

  // One row of the ORF track: a chevron per codon intersecting the row, the
  // single-letter AA shown on the segment holding the codon's middle base.
  Viewer.prototype.renderAaTrack = function (rowStart, rowEnd, cw) {
    var codons = this.computeCodons();
    if (!codons.length) { return ''; }
    var out = [];
    for (var i = 0; i < codons.length; i++) {
      var c = codons[i];
      if (c.end < rowStart || c.start > rowEnd) { continue; }
      var segStart = Math.max(c.start, rowStart), segEnd = Math.min(c.end, rowEnd);
      var left = (segStart - rowStart) * cw, width = (segEnd - segStart + 1) * cw;
      var mid = c.start + 1, showLetter = (mid >= segStart && mid <= segEnd);
      var bg = AA_COLORS[c.aa] || '#cfcfcf';
      var cls = 'oriv-aa ' + (c.strand === -1 ? 'oriv-aa-rev' : 'oriv-aa-fwd') +
                (c.aa === '*' ? ' oriv-aa-stop' : '');
      out.push('<span class="' + cls + '" data-fi="' + c.fi + '" style="left:' + left.toFixed(2) + 'px;width:' +
        width.toFixed(2) + 'px;background:' + bg + ';color:' + idealText(bg) + '" title="' +
        esc(c.aa + ' @ ' + c.start + '-' + c.end + ' · right-click to copy ORF') + '">' + (showLetter ? esc(c.aa) : '') + '</span>');
    }
    if (!out.length) { return ''; }
    return '<div class="oriv-aa-track">' + out.join('') + '</div>';
  };

  // Extract a feature's nucleotide sequence (spliced across spans; reverse-strand
  // features returned as their coding strand) and its translation.
  Viewer.prototype.featDna = function (f) {
    var s = '', seq = this.seq;
    for (var i = 0; i < f.spans.length; i++) {
      var a = f.spans[i].start, b = f.spans[i].end;
      if (a <= b) { s += seq.slice(a - 1, b); }
      else { s += seq.slice(a - 1) + seq.slice(0, b); } // origin wrap
    }
    return (f.strand === -1) ? revComp(s) : s;
  };
  Viewer.prototype.featAa = function (f) {
    var dna = this.featDna(f), aa = '';
    for (var i = 0; i + 3 <= dna.length; i += 3) { aa += translateCodon(dna.substr(i, 3)); }
    return aa;
  };

  // Right-click context menu: copy a feature's / ORF's DNA, protein, FASTA, or name.
  Viewer.prototype.showMenu = function (e, f) {
    var self = this;
    if (!this.menu) {
      this.menu = document.createElement('div');
      this.menu.className = 'oriv-menu';
      document.body.appendChild(this.menu);
    }
    var menu = this.menu;
    var name = featureLabel(f);
    var rngTxt = (f.start === f.end ? String(f.start) : (f.start + '-' + f.end)) + (f.strand === -1 ? ' rev' : '');
    var items = [{ label: 'Copy DNA sequence', get: function () { return self.featDna(f); } }];
    if (f.type === 'CDS') { items.push({ label: 'Copy protein (amino acids)', get: function () { return self.featAa(f); } }); }
    items.push({ label: 'Copy as FASTA', get: function () { return toFasta(name + ' [' + f.type + ' ' + rngTxt + ']', self.featDna(f)); } });
    items.push({ label: 'Copy feature name', get: function () { return name; } });

    var html = '<div class="oriv-menu-head">' + esc(name) + ' · ' + esc(f.type) + ' ' + esc(rngTxt) + '</div>';
    items.forEach(function (it, idx) { html += '<button class="oriv-menu-item" data-mi="' + idx + '">' + esc(it.label) + '</button>'; });
    menu.innerHTML = html;

    menu.style.display = 'block';
    var pad = 8, x = e.clientX + 2, y = e.clientY + 2, w = menu.offsetWidth, h = menu.offsetHeight;
    if (x + w + pad > window.innerWidth) { x = window.innerWidth - w - pad; }
    if (y + h + pad > window.innerHeight) { y = window.innerHeight - h - pad; }
    menu.style.left = Math.max(pad, x) + 'px';
    menu.style.top = Math.max(pad, y) + 'px';

    function hide() {
      menu.style.display = 'none';
      document.removeEventListener('click', onDoc, true);
      document.removeEventListener('keydown', onKey, true);
      window.removeEventListener('scroll', hide, true);
    }
    function onDoc(ev) { if (!menu.contains(ev.target)) { hide(); } }
    function onKey(ev) { if (ev.key === 'Escape') { hide(); } }

    Array.prototype.forEach.call(menu.querySelectorAll('.oriv-menu-item'), function (btn) {
      btn.addEventListener('click', function () {
        var text = items[parseInt(btn.getAttribute('data-mi'), 10)].get();
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(
            function () { btn.textContent = '✓ Copied'; setTimeout(hide, 550); },
            function () { btn.textContent = 'Copy failed'; }
          );
        } else { btn.textContent = 'Clipboard unavailable'; }
      });
    });
    setTimeout(function () {
      document.addEventListener('click', onDoc, true);
      document.addEventListener('keydown', onKey, true);
      window.addEventListener('scroll', hide, true);
    }, 0);
  };

  // Expand the panel into a near-fullscreen overlay (and back). Reflows the
  // sequence to the wider width afterwards.
  Viewer.prototype.toggleExpand = function (btn) {
    var self = this;
    this.expanded = !this.expanded;
    if (btn) { this._expandBtn = btn; }
    var label = this._expandBtn ? this._expandBtn.querySelector('.oriv-exp-label') : null;

    if (this.expanded) {
      if (!this._backdrop) {
        this._backdrop = document.createElement('div');
        this._backdrop.className = 'oriv-backdrop';
        document.body.appendChild(this._backdrop);
        this._backdrop.addEventListener('click', function () { self.toggleExpand(self._expandBtn); });
      }
      this._backdrop.style.display = 'block';
      void this._backdrop.offsetWidth; // reflow so the opacity transition runs
      this._backdrop.classList.add('show');
      this.panel.classList.add('oriv-expanded');
      this._escHandler = function (e) { if (e.key === 'Escape') { self.toggleExpand(self._expandBtn); } };
      document.addEventListener('keydown', this._escHandler);
      if (label) { label.textContent = 'Collapse'; }
    } else {
      this.panel.classList.remove('oriv-expanded');
      if (this._backdrop) {
        var bd = this._backdrop;
        bd.classList.remove('show');
        setTimeout(function () { bd.style.display = 'none'; }, 260);
      }
      if (this._escHandler) { document.removeEventListener('keydown', this._escHandler); this._escHandler = null; }
      if (label) { label.textContent = 'Expand'; }
    }
    // Reflow bases-per-row to the new width (ResizeObserver also catches this).
    setTimeout(function () { self.relayout(); }, 30);
  };

  // ── Responsive layout: pick bases-per-row from the measured panel width ──
  Viewer.prototype._measureCharW = function () {
    if (this._charW) { return this._charW; }
    var probe = document.createElement('span');
    probe.className = 'oriv-line oriv-fwd';
    probe.style.position = 'absolute'; probe.style.visibility = 'hidden';
    probe.innerHTML = '<span class="oriv-base">G</span>';
    this.seqEl.appendChild(probe);
    var w = probe.firstChild.getBoundingClientRect().width || 8;
    this.seqEl.removeChild(probe);
    this._charW = w;
    return w;
  };

  Viewer.prototype.computePerRow = function () {
    var cw = this._measureCharW();
    var cs = getComputedStyle(this.seqEl);
    var avail = this.seqEl.clientWidth - (parseFloat(cs.paddingLeft) || 0) - (parseFloat(cs.paddingRight) || 0);
    var gutter = 6 * cw + 10 + 2; // .oriv-pos width(6ch) + padding-right + slack
    var per = Math.floor((avail - gutter) / cw); // fill the row exactly (ruler ticks are absolute, so any width is fine)
    if (!isFinite(per) || per < 10) { return 60; }
    return Math.max(20, Math.min(200, per));
  };

  Viewer.prototype.relayout = function () {
    if (!this.L) { return; }
    var per = this.computePerRow();
    if (per !== this.PER || !this._rendered) {
      this.PER = per; this._rendered = true;
      this.renderSequence();
      if (this.searchEl && this.searchEl.value.trim()) { this.runSearch(); }
    }
  };

  Viewer.prototype.positionsFor = function (m) {
    var ps = [];
    if (m.start <= m.end) {
      for (var p = m.start; p <= m.end; p++) { ps.push(p); }
    } else { // wrapped match
      for (var p1 = m.start; p1 <= this.L; p1++) { ps.push(p1); }
      for (var p2 = 1; p2 <= m.end; p2++) { ps.push(p2); }
    }
    return ps;
  };

  Viewer.prototype.clearHits = function () {
    var hits = this.seqEl.querySelectorAll('.oriv-hit');
    for (var i = 0; i < hits.length; i++) {
      hits[i].classList.remove('oriv-hit');
      hits[i].classList.remove('active');
    }
  };

  Viewer.prototype.findAll = function (needle) {
    var res = [];
    if (!needle) { return res; }
    var hay = this.seq;
    var extended = (this.gb.topology === 'circular' && needle.length > 1)
      ? hay + hay.slice(0, needle.length - 1) : hay;
    var from = 0;
    while (true) {
      var idx = extended.indexOf(needle, from);
      if (idx < 0) { break; }
      var start = idx + 1;
      var end = idx + needle.length;
      if (end > this.L) { end = end - this.L; } // wrapped past origin
      res.push({ start: start, end: end, kind: 'dna' });
      from = idx + 1;
    }
    return res;
  };

  Viewer.prototype.runSearch = function () {
    this.clearHits();
    this.matches = [];
    this.activeIdx = -1;
    var q = (this.searchEl.value || '').trim().toUpperCase();
    if (q.length) {
      var isDna = /^[ACGTUNRYSWKMBDHV]+$/.test(q);
      if (isDna) {
        var fwd = this.findAll(q);
        for (var a = 0; a < fwd.length; a++) { this.matches.push(fwd[a]); }
        var rc = revComp(q);
        if (rc !== q) {
          var rev = this.findAll(rc);
          for (var b = 0; b < rev.length; b++) { rev[b].kind = 'dna-rc'; this.matches.push(rev[b]); }
        }
      }
      // feature-name / type search -> highlight whole feature span(s)
      var feats = this.gb.features;
      for (var fi = 0; fi < feats.length; fi++) {
        var f = feats[fi];
        if (f.type === 'source') { continue; }
        var hay = [f.type];
        var q2 = f.qualifiers || {};
        ['label', 'gene', 'product', 'note', 'standard_name'].forEach(function (k) {
          if (q2[k]) { hay = hay.concat(q2[k]); }
        });
        var match = false;
        for (var h = 0; h < hay.length; h++) {
          if (String(hay[h]).toUpperCase().indexOf(q) !== -1) { match = true; break; }
        }
        if (match) {
          for (var sp = 0; sp < f.spans.length; sp++) {
            this.matches.push({ start: f.spans[sp].start, end: f.spans[sp].end, kind: 'feat' });
          }
        }
      }
      // sort + dedupe
      this.matches.sort(function (x, y) { return (x.start - y.start) || (x.end - y.end); });
      var dedup = [], seen = {};
      for (var d = 0; d < this.matches.length; d++) {
        var key = this.matches[d].start + ':' + this.matches[d].end;
        if (!seen[key]) { seen[key] = 1; dedup.push(this.matches[d]); }
      }
      this.matches = dedup;
      // paint
      for (var m = 0; m < this.matches.length; m++) {
        var ps = this.positionsFor(this.matches[m]);
        for (var pi = 0; pi < ps.length; pi++) {
          var node = this.charMap[ps[pi]];
          if (node) { node.classList.add('oriv-hit'); }
        }
      }
      if (this.matches.length) { this.activeIdx = 0; this.paintActive(true); }
    }
    this.updateCount();
  };

  Viewer.prototype.paintActive = function (scroll) {
    var prev = this.seqEl.querySelectorAll('.oriv-hit.active');
    for (var i = 0; i < prev.length; i++) { prev[i].classList.remove('active'); }
    if (this.activeIdx < 0 || this.activeIdx >= this.matches.length) { return; }
    var ps = this.positionsFor(this.matches[this.activeIdx]);
    var first = null;
    for (var p = 0; p < ps.length; p++) {
      var node = this.charMap[ps[p]];
      if (node) { node.classList.add('active'); if (!first) { first = node; } }
    }
    if (scroll && first) { first.scrollIntoView({ block: 'center', behavior: 'smooth' }); }
  };

  Viewer.prototype.step = function (delta) {
    if (!this.matches.length) { return; }
    this.activeIdx = (this.activeIdx + delta + this.matches.length) % this.matches.length;
    this.paintActive(true);
    this.updateCount();
  };

  Viewer.prototype.updateCount = function () {
    if (!this.matches.length) {
      this.countEl.textContent = (this.searchEl.value.trim() ? '0 / 0' : '');
    } else {
      this.countEl.textContent = (this.activeIdx + 1) + ' / ' + this.matches.length;
    }
    this.prevBtn.disabled = this.matches.length < 2;
    this.nextBtn.disabled = this.matches.length < 2;
  };

  Viewer.prototype.jumpToFeature = function (f) {
    var node = this.charMap[f.start];
    if (node) {
      node.scrollIntoView({ block: 'center', behavior: 'smooth' });
      var ps = this.positionsFor({ start: f.spans[0].start, end: f.spans[0].end });
      for (var p = 0; p < ps.length; p++) {
        var el = this.charMap[ps[p]];
        if (el) {
          el.classList.add('oriv-flash');
          (function (e) { setTimeout(function () { e.classList.remove('oriv-flash'); }, 1100); })(el);
        }
      }
    }
  };

  Viewer.prototype.build = function () {
    var self = this;
    var outer = document.createElement('div');
    outer.className = 'msg assistant';

    var topology = this.gb.topology === 'circular' ? 'circular' : 'linear';
    var nFeat = this.featList.length;
    var resist = detectResistance(this.featList);
    var resistHtml = resist.length
      ? '<span class="oriv-resist" title="Antibiotic resistance marker(s)">' + ICON_PILL + esc(resist.join(', ')) + '</span>'
      : '';

    var panel = document.createElement('div');
    panel.className = 'msg-bubble-assistant oriv-panel';
    panel.innerHTML =
      '<div class="oriv-head">' +
        '<span class="oriv-title">' + esc(this.name) + '</span>' +
        '<span class="oriv-meta">' +
          '<span><b>' + this.L.toLocaleString() + '</b> bp</span>' +
          '<span>' + topology + '</span>' +
          '<span><b>' + nFeat + '</b> features</span>' +
          resistHtml +
        '</span>' +
        '<span class="oriv-spacer"></span>' +
        '<label class="oriv-toggle"><input type="checkbox" data-role="orf"> reading frame</label>' +
        '<label class="oriv-toggle"><input type="checkbox" data-role="comp"> complement</label>' +
        '<button class="oriv-mini-btn" data-role="copy">Copy DNA</button>' +
        '<button class="oriv-mini-btn oriv-icon-btn" data-role="dl-gb" title="Download GenBank (.gb) — identical to the main window">' + ICON_DL + '.gb</button>' +
        '<button class="oriv-mini-btn oriv-icon-btn" data-role="dl-fasta" title="Download FASTA (.fasta)">' + ICON_DL + '.fasta</button>' +
        '<button class="oriv-mini-btn oriv-icon-btn" data-role="expand" title="Expand to a wider view">' +
          '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>' +
          '<span class="oriv-exp-label">Expand</span>' +
        '</button>' +
        '<span class="oriv-search-wrap">' +
          '<input class="oriv-search" type="text" placeholder="Find DNA or feature…" data-role="search">' +
          '<button class="oriv-nav" data-role="prev" title="Previous match" disabled>&#8593;</button>' +
          '<button class="oriv-nav" data-role="next" title="Next match" disabled>&#8595;</button>' +
          '<span class="oriv-count" data-role="count"></span>' +
        '</span>' +
      '</div>' +
      '<div class="oriv-legend" data-role="legend"></div>' +
      '<div class="oriv-seq" data-role="seq"></div>';

    outer.appendChild(panel);
    this.el = outer;
    this.panel = panel;
    this.seqEl = panel.querySelector('[data-role="seq"]');
    this.searchEl = panel.querySelector('[data-role="search"]');
    this.countEl = panel.querySelector('[data-role="count"]');
    this.prevBtn = panel.querySelector('[data-role="prev"]');
    this.nextBtn = panel.querySelector('[data-role="next"]');

    // legend chips (skip source; cap to keep it tidy)
    var legend = panel.querySelector('[data-role="legend"]');
    var shown = this.gb.features.filter(function (f) { return f.type !== 'source'; });
    for (var i = 0; i < shown.length; i++) {
      (function (f) {
        var chip = document.createElement('span');
        chip.className = 'oriv-chip';
        var rng = (f.start === f.end) ? String(f.start) : (f.start + '–' + f.end);
        chip.innerHTML = '<span class="dot" style="background:' + featureColor(f) + '"></span>' +
          '<span class="nm">' + esc(featureLabel(f)) + '</span>' +
          '<span class="rg">' + (f.strand === -1 ? '← ' : '') + rng + '</span>';
        chip.title = f.type + ' · ' + rng + (f.strand === -1 ? ' (reverse)' : '');
        chip.addEventListener('click', function () { self.jumpToFeature(f); });
        legend.appendChild(chip);
      })(shown[i]);
    }

    // events
    panel.querySelector('[data-role="comp"]').addEventListener('change', function () {
      self.showComp = this.checked;
      self.renderSequence();
      self.runSearch();
    });
    panel.querySelector('[data-role="orf"]').addEventListener('change', function () {
      self.showOrf = this.checked;
      self.renderSequence();
      self.runSearch();
    });
    panel.querySelector('[data-role="copy"]').addEventListener('click', function () {
      var btn = this;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(self.seq).then(function () {
          var t = btn.textContent; btn.textContent = 'Copied!';
          setTimeout(function () { btn.textContent = t; }, 1200);
        });
      }
    });
    panel.querySelector('[data-role="expand"]').addEventListener('click', function () {
      self.toggleExpand(this);
    });
    panel.querySelector('[data-role="dl-gb"]').addEventListener('click', function () {
      var base = (self.name || 'construct').replace(/[^\w.\-]+/g, '_').replace(/\.(gb|gbk)$/i, '');
      downloadText(base + '.gb', self.rawGb || '', 'chemical/x-genbank');
    });
    panel.querySelector('[data-role="dl-fasta"]').addEventListener('click', function () {
      var base = (self.name || 'construct').replace(/[^\w.\-]+/g, '_').replace(/\.(fa|fasta)$/i, '');
      downloadText(base + '.fasta', toFasta(self.name || 'construct', self.seq), 'text/x-fasta');
    });
    this.searchEl.addEventListener('input', function () {
      clearTimeout(self.searchTimer);
      self.searchTimer = setTimeout(function () { self.runSearch(); }, 110);
    });
    this.searchEl.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { e.preventDefault(); self.step(e.shiftKey ? -1 : 1); }
    });
    this.prevBtn.addEventListener('click', function () { self.step(-1); });
    this.nextBtn.addEventListener('click', function () { self.step(1); });

    // ── Hover tooltip: feature name(s) at the hovered base ──────────────────
    this.tip = document.createElement('div');
    this.tip.className = 'oriv-tip';
    this.tip.style.display = 'none';
    document.body.appendChild(this.tip);
    this._tipPos = -1;

    function tipHtml(pos) {
      var fs = self.featAt ? self.featAt[pos] : null;
      if (!fs || !fs.length) { return ''; }
      var sorted = fs.slice().sort(function (a, b) { return (a.end - a.start) - (b.end - b.start); });
      var h = '<div class="oriv-tip-pos">position ' + pos + '</div>';
      for (var i = 0; i < sorted.length && i < 6; i++) {
        var f = sorted[i];
        var rng = (f.start === f.end) ? String(f.start) : (f.start + '–' + f.end);
        var arrow = f.strand === -1 ? ' ◀' : (f.strand === 1 ? ' ▶' : '');
        h += '<div class="oriv-tip-row">' +
          '<span class="oriv-tip-dot" style="background:' + featureColor(f) + '"></span>' +
          '<span class="oriv-tip-nm">' + esc(featureLabel(f)) + '</span>' +
          '<span class="oriv-tip-ty">' + esc(f.type) + arrow + ' ' + rng + '</span>' +
          '</div>';
      }
      if (sorted.length > 6) { h += '<div class="oriv-tip-more">+' + (sorted.length - 6) + ' more</div>'; }
      return h;
    }
    function placeTip(e) {
      var pad = 10, x = e.clientX + 14, y = e.clientY + 16;
      var w = self.tip.offsetWidth, ht = self.tip.offsetHeight;
      if (x + w + pad > window.innerWidth) { x = e.clientX - w - 14; }
      if (y + ht + pad > window.innerHeight) { y = e.clientY - ht - 16; }
      if (x < pad) { x = pad; }
      if (y < pad) { y = pad; }
      self.tip.style.left = x + 'px';
      self.tip.style.top = y + 'px';
    }
    function hideTip() { self.tip.style.display = 'none'; self._tipPos = -1; }

    this.seqEl.addEventListener('mousemove', function (e) {
      var t = e.target;
      if (t && t.classList && t.classList.contains('oriv-base')) {
        var pos = parseInt(t.getAttribute('data-pos'), 10);
        if (self.featAt && self.featAt[pos] && self.featAt[pos].length) {
          if (self._tipPos !== pos) { self.tip.innerHTML = tipHtml(pos); self._tipPos = pos; }
          self.tip.style.display = 'block';
          placeTip(e);
          return;
        }
      }
      hideTip();
    });
    this.seqEl.addEventListener('mouseleave', hideTip);

    // Right-click a feature bar, ORF chevron, or highlighted base → copy menu.
    this.seqEl.addEventListener('contextmenu', function (e) {
      var f = null;
      var hit = e.target.closest ? e.target.closest('.oriv-feat-bar, .oriv-aa') : null;
      if (hit && hit.getAttribute('data-fi') !== null) {
        f = self.featList[parseInt(hit.getAttribute('data-fi'), 10)];
      } else if (e.target.classList && e.target.classList.contains('oriv-base')) {
        var pos = parseInt(e.target.getAttribute('data-pos'), 10);
        var fs = self.featAt && self.featAt[pos];
        if (fs && fs.length) {
          f = fs.slice().sort(function (a, b) { return (a.end - a.start) - (b.end - b.start); })[0];
        }
      }
      if (!f) { return; } // nothing here — leave the browser's native menu
      e.preventDefault();
      self.showMenu(e, f);
    });

    // Responsive: recompute bases-per-row when the panel width changes.
    if (window.ResizeObserver) {
      var roTimer = null;
      this._ro = new ResizeObserver(function () {
        clearTimeout(roTimer);
        roTimer = setTimeout(function () { self.relayout(); }, 150);
      });
      this._ro.observe(this.panel);
    }

    if (!this.L) {
      this.seqEl.innerHTML = '<div class="oriv-empty">No sequence found in this construct.</div>';
    }
    // The sequence is rendered by relayout() after the panel is attached to the
    // DOM (we need its measured width to choose bases-per-row).
    return outer;
  };

  // ── Public API ───────────────────────────────────────────────────────────
  window.OriViewer = {
    open: function (genbankText, name, btn) {
      // toggle if this button already opened a panel
      if (btn && btn._orivPanel) {
        var p = btn._orivPanel;
        var hidden = (p.style.display === 'none');
        p.style.display = hidden ? '' : 'none';
        var lbl = btn.querySelector('.viewer-btn-label');
        if (lbl) { lbl.textContent = hidden ? 'Hide Viewer' : 'Open in Viewer'; }
        if (hidden) { p.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }
        return p;
      }
      var gb = parseGenBank(genbankText);
      var v = new Viewer(gb, name, genbankText);
      var el = v.build();
      var host = btn ? btn.closest('.msg.assistant') : null;
      if (host && host.parentNode) {
        host.parentNode.insertBefore(el, host.nextSibling);
      } else {
        var inner = document.querySelector('.messages-inner') || document.body;
        inner.appendChild(el);
      }
      v.relayout(); // render the sequence now that the panel has a measurable width
      window.OriViewer._last = v; // most-recently-opened viewer (debug/test handle)
      if (btn) {
        btn._orivPanel = el;
        var lbl2 = btn.querySelector('.viewer-btn-label');
        if (lbl2) { lbl2.textContent = 'Hide Viewer'; }
      }
      el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      return el;
    },
    // exposed for debugging / reuse
    _parseGenBank: parseGenBank
  };
})();
