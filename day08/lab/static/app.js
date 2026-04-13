/**
 * app.js — RAG Pipeline Lab UI
 * =============================
 * Client-side logic for the RAG Pipeline testing interface.
 */

// =============================================================================
// TAB NAVIGATION
// =============================================================================

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        // Deactivate all tabs
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

        // Activate clicked tab
        btn.classList.add('active');
        const tabId = btn.dataset.tab;
        document.getElementById(`panel-${tabId}`).classList.add('active');

        // Load data for specific tabs
        if (tabId === 'index') {
            loadIndexStatus();
            loadChunks();
        } else if (tabId === 'eval') {
            loadTestQuestions();
        }
    });
});


// =============================================================================
// STARTUP
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    loadIndexStatus();
});


// =============================================================================
// CHAT
// =============================================================================

let chatHistory = [];

function fillQuery(text) {
    document.getElementById('chat-input').value = text;
    document.getElementById('chat-input').focus();
}

async function sendChat(e) {
    e.preventDefault();

    const input = document.getElementById('chat-input');
    const query = input.value.trim();
    if (!query) return;

    const sendBtn = document.getElementById('send-btn');
    sendBtn.disabled = true;

    // Add user message
    addMessage('user', query);
    input.value = '';

    // Remove welcome card
    const welcome = document.querySelector('.welcome-card');
    if (welcome) welcome.remove();

    // Show typing indicator
    const typingId = showTyping();

    // Get config
    const config = {
        query,
        retrieval_mode: document.getElementById('chat-retrieval-mode').value,
        top_k_search: parseInt(document.getElementById('chat-top-k-search').value),
        top_k_select: parseInt(document.getElementById('chat-top-k-select').value),
        use_rerank: document.getElementById('chat-rerank').checked,
    };

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });

        removeTyping(typingId);

        if (!response.ok) {
            const err = await response.json();
            addMessage('bot', `❌ Error: ${err.error || 'Unknown error'}`, []);
            return;
        }

        const data = await response.json();

        // Add bot message with metadata
        const meta = [
            `mode: ${data.config.retrieval_mode}`,
            `chunks: ${data.chunks.length}`,
            `rerank: ${data.config.use_rerank ? 'on' : 'off'}`,
        ];
        addMessage('bot', data.answer, meta);

        // Update sidebar with chunks
        updateSidebarChunks(data.chunks);

    } catch (err) {
        removeTyping(typingId);
        addMessage('bot', `❌ Network error: ${err.message}`, []);
    } finally {
        sendBtn.disabled = false;
        document.getElementById('chat-input').focus();
    }
}

function addMessage(role, text, metaTags = []) {
    const container = document.getElementById('chat-messages');

    if (role === 'user') {
        const div = document.createElement('div');
        div.className = 'message message-user';
        div.innerHTML = `<div class="message-bubble">${escapeHtml(text)}</div>`;
        container.appendChild(div);
    } else {
        const div = document.createElement('div');
        div.className = 'message message-bot';

        let metaHtml = '';
        if (metaTags.length > 0) {
            metaHtml = `<div class="message-meta">
                ${metaTags.map(t => `<span class="meta-tag">${t}</span>`).join('')}
            </div>`;
        }

        div.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-bubble">
                ${formatAnswer(text)}
                ${metaHtml}
            </div>
        `;
        container.appendChild(div);
    }

    container.scrollTop = container.scrollHeight;
}

function formatAnswer(text) {
    // Convert citation markers [1], [2] to styled spans
    let formatted = escapeHtml(text);
    formatted = formatted.replace(/\[(\d+)\]/g, '<strong style="color: #a78bfa;">[$1]</strong>');
    // Convert newlines
    formatted = formatted.replace(/\n/g, '<br>');
    return formatted;
}

let typingCounter = 0;

function showTyping() {
    const id = `typing-${++typingCounter}`;
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'message message-bot';
    div.id = id;
    div.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-bubble">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeTyping(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function updateSidebarChunks(chunks) {
    const container = document.getElementById('sidebar-chunks');
    const badge = document.getElementById('chunk-count');
    badge.textContent = chunks.length;

    if (chunks.length === 0) {
        container.innerHTML = '<div class="sidebar-empty"><p>No chunks retrieved.</p></div>';
        return;
    }

    container.innerHTML = chunks.map(chunk => `
        <div class="chunk-card" onclick="this.querySelector('.chunk-text').classList.toggle('expanded')">
            <div class="chunk-card-header">
                <span class="chunk-index">[${chunk.index}]</span>
                <span class="chunk-score">score: ${chunk.score}</span>
            </div>
            <div class="chunk-source">${extractFilename(chunk.source)}</div>
            ${chunk.section ? `<span class="chunk-section-tag">${chunk.section}</span>` : ''}
            <div class="chunk-text">${escapeHtml(chunk.text)}</div>
        </div>
    `).join('');
}


// =============================================================================
// COMPARE STRATEGIES
// =============================================================================

function fillCompare(text) {
    document.getElementById('compare-input').value = text;
    document.getElementById('compare-input').focus();
}

async function runComparison(e) {
    e.preventDefault();

    const input = document.getElementById('compare-input');
    const query = input.value.trim();
    if (!query) return;

    const btn = document.getElementById('compare-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Comparing...';

    const resultsContainer = document.getElementById('compare-results');
    resultsContainer.innerHTML = `
        <div class="compare-empty" style="grid-column: 1 / -1;">
            <div class="spinner spinner-dark" style="width: 32px; height: 32px; border-width: 3px;"></div>
            <p style="margin-top: 12px;">Running all three strategies...</p>
        </div>
    `;

    try {
        const response = await fetch('/api/retrieval/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query }),
        });

        const data = await response.json();

        if (data.error) {
            resultsContainer.innerHTML = `<div class="compare-empty" style="grid-column: 1 / -1;"><p>❌ ${data.error}</p></div>`;
            return;
        }

        resultsContainer.innerHTML = ['dense', 'sparse', 'hybrid'].map(mode => {
            const result = data.results[mode];
            if (result.error) {
                return `
                    <div class="compare-card">
                        <div class="compare-card-header">
                            <h4>${modeLabel(mode)}</h4>
                            <span class="strategy-badge ${mode}">${mode}</span>
                        </div>
                        <div class="compare-card-body">
                            <p style="color: var(--accent-red);">❌ ${result.error}</p>
                        </div>
                    </div>
                `;
            }

            return `
                <div class="compare-card">
                    <div class="compare-card-header">
                        <h4>${modeLabel(mode)}</h4>
                        <span class="strategy-badge ${mode}">${mode}</span>
                    </div>
                    <div class="compare-card-body">
                        <div class="compare-answer">${formatAnswer(result.answer)}</div>
                        <div class="compare-chunks-title">Retrieved Chunks (${result.chunks.length})</div>
                        ${result.chunks.map(c => `
                            <div class="compare-chunk-item">
                                ${escapeHtml(c.text)}
                                <div class="compare-chunk-meta">
                                    ${extractFilename(c.source)} | ${c.section} | score: ${c.score}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }).join('');

    } catch (err) {
        resultsContainer.innerHTML = `<div class="compare-empty" style="grid-column: 1 / -1;"><p>❌ Network error: ${err.message}</p></div>`;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Compare';
    }
}

function modeLabel(mode) {
    const labels = { dense: '🧠 Dense (Embedding)', sparse: '🔤 Sparse (BM25)', hybrid: '⚡ Hybrid (RRF)' };
    return labels[mode] || mode;
}


// =============================================================================
// INDEX EXPLORER
// =============================================================================

async function loadIndexStatus() {
    try {
        const response = await fetch('/api/index/status');
        const data = await response.json();

        // Update header status
        const statusEl = document.getElementById('header-status');
        if (data.total_chunks > 0) {
            statusEl.innerHTML = `<span class="status-dot online"></span><span class="status-text">${data.total_chunks} chunks indexed</span>`;
        } else {
            statusEl.innerHTML = `<span class="status-dot"></span><span class="status-text">No index</span>`;
        }

        // Update stats
        document.getElementById('stat-chunks').textContent = data.total_chunks || 0;
        document.getElementById('stat-docs').textContent = (data.doc_files || []).length;
        document.getElementById('stat-departments').textContent = Object.keys(data.departments || {}).length;
        document.getElementById('stat-sections').textContent = Object.keys(data.sections || {}).length;

        // Document list
        const docList = document.getElementById('doc-list');
        if (data.doc_files && data.doc_files.length > 0) {
            docList.innerHTML = data.doc_files.map(f => `
                <div class="detail-item">
                    <span class="detail-item-name">📄 ${f}</span>
                </div>
            `).join('');
        }

        // Department distribution
        const deptDist = document.getElementById('dept-dist');
        if (data.departments) {
            deptDist.innerHTML = Object.entries(data.departments).map(([dept, count]) => `
                <div class="detail-item">
                    <span class="detail-item-name">${dept}</span>
                    <span class="detail-item-count">${count} chunks</span>
                </div>
            `).join('');
        }

    } catch (err) {
        console.error('Failed to load index status:', err);
    }
}

async function loadChunks() {
    const limit = document.getElementById('chunk-limit')?.value || 20;
    const wrap = document.getElementById('chunks-table-wrap');

    try {
        const response = await fetch(`/api/index/chunks?limit=${limit}`);
        const data = await response.json();

        if (!data.chunks || data.chunks.length === 0) {
            wrap.innerHTML = '<p class="text-muted text-center" style="padding: 20px;">No chunks found. Build index first.</p>';
            return;
        }

        wrap.innerHTML = `
            <table class="chunks-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Source</th>
                        <th>Section</th>
                        <th>Department</th>
                        <th>Date</th>
                        <th>Text Preview</th>
                    </tr>
                </thead>
                <tbody>
                    ${data.chunks.map(c => `
                        <tr>
                            <td class="meta-col">${c.index}</td>
                            <td class="meta-col">${extractFilename(c.source)}</td>
                            <td class="meta-col">${c.section}</td>
                            <td class="meta-col">${c.department}</td>
                            <td class="meta-col">${c.effective_date}</td>
                            <td class="text-col" title="${escapeAttr(c.text)}">${escapeHtml(c.text)}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;

    } catch (err) {
        wrap.innerHTML = `<p class="text-muted text-center" style="padding: 20px;">Error loading chunks: ${err.message}</p>`;
    }
}

async function rebuildIndex() {
    if (!confirm('This will rebuild the entire index. Continue?')) return;

    const btn = document.getElementById('rebuild-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Rebuilding...';

    try {
        const response = await fetch('/api/index/rebuild', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            alert('✓ Index rebuilt successfully!');
            loadIndexStatus();
            loadChunks();
        } else {
            alert('Error: ' + (data.error || 'Unknown error'));
        }

    } catch (err) {
        alert('Network error: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Rebuild Index';
    }
}


// =============================================================================
// EVALUATION
// =============================================================================

async function loadTestQuestions() {
    const container = document.getElementById('test-questions-list');

    try {
        const response = await fetch('/api/eval/questions');
        const data = await response.json();

        if (!data.questions || data.questions.length === 0) {
            container.innerHTML = '<p class="text-muted">No test questions found.</p>';
            return;
        }

        container.innerHTML = data.questions.map(q => `
            <div class="tq-card">
                <div class="tq-card-header">
                    <span class="tq-id">${q.id}</span>
                    <span class="tq-category">${q.category}</span>
                </div>
                <div class="tq-question">${escapeHtml(q.question)}</div>
                <div class="tq-expected">Expected: ${escapeHtml(q.expected_answer).substring(0, 120)}...</div>
                <span class="tq-difficulty ${q.difficulty}">${q.difficulty}</span>
            </div>
        `).join('');

    } catch (err) {
        container.innerHTML = `<p class="text-muted">Error: ${err.message}</p>`;
    }
}

let evalResults = {};

async function runEval(configType) {
    const btnId = configType === 'baseline' ? 'eval-baseline-btn' : 'eval-variant-btn';
    const btn = document.getElementById(btnId);
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Running ${configType}...`;

    const resultsContainer = document.getElementById('eval-results');
    resultsContainer.innerHTML = `
        <div class="eval-empty">
            <div class="spinner spinner-dark" style="width: 36px; height: 36px; border-width: 3px;"></div>
            <p style="margin-top: 16px;">Running LLM-as-Judge evaluation for <strong>${configType}</strong>...<br>This may take a minute.</p>
        </div>
    `;

    try {
        const response = await fetch('/api/eval/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config: configType }),
        });

        const data = await response.json();

        if (data.error) {
            resultsContainer.innerHTML = `<div class="eval-empty"><p>❌ ${data.error}</p></div>`;
            return;
        }

        evalResults[configType] = data;
        renderEvalResults(configType, data);

    } catch (err) {
        resultsContainer.innerHTML = `<div class="eval-empty"><p>❌ Network error: ${err.message}</p></div>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = configType === 'baseline'
            ? '🏁 Run Baseline (Dense)'
            : '⚡ Run Variant (Hybrid)';
    }
}

function renderEvalResults(label, data) {
    const container = document.getElementById('eval-results');
    const avg = data.averages;

    function scoreClass(val) {
        if (val === null || val === undefined) return '';
        if (val >= 4) return 'good';
        if (val >= 3) return 'mid';
        return 'bad';
    }

    function scoreCellClass(val) {
        if (val === null || val === undefined) return '';
        return `s${val}`;
    }

    container.innerHTML = `
        <div class="eval-summary">
            <h3>📊 Results: ${label === 'baseline' ? 'Baseline (Dense)' : 'Variant (Hybrid)'}</h3>
            <div class="eval-avg-grid">
                <div class="eval-avg-card">
                    <div class="eval-avg-value ${scoreClass(avg.faithfulness)}">${avg.faithfulness ?? 'N/A'}</div>
                    <div class="eval-avg-label">Faithfulness</div>
                </div>
                <div class="eval-avg-card">
                    <div class="eval-avg-value ${scoreClass(avg.relevance)}">${avg.relevance ?? 'N/A'}</div>
                    <div class="eval-avg-label">Relevance</div>
                </div>
                <div class="eval-avg-card">
                    <div class="eval-avg-value ${scoreClass(avg.context_recall)}">${avg.context_recall ?? 'N/A'}</div>
                    <div class="eval-avg-label">Context Recall</div>
                </div>
                <div class="eval-avg-card">
                    <div class="eval-avg-value ${scoreClass(avg.completeness)}">${avg.completeness ?? 'N/A'}</div>
                    <div class="eval-avg-label">Completeness</div>
                </div>
            </div>
        </div>
        <div class="eval-detail-table">
            <table class="eval-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Category</th>
                        <th>Question</th>
                        <th>Answer (Preview)</th>
                        <th>Faith.</th>
                        <th>Relev.</th>
                        <th>Recall</th>
                        <th>Comp.</th>
                    </tr>
                </thead>
                <tbody>
                    ${data.results.map(r => `
                        <tr>
                            <td class="meta-col">${r.id}</td>
                            <td class="meta-col">${r.category}</td>
                            <td style="max-width: 200px;">${escapeHtml(r.query).substring(0, 50)}...</td>
                            <td style="max-width: 250px; font-size: 0.78rem;">${escapeHtml(r.answer).substring(0, 80)}...</td>
                            <td class="score-cell ${scoreCellClass(r.faithfulness)}">${r.faithfulness ?? '—'}</td>
                            <td class="score-cell ${scoreCellClass(r.relevance)}">${r.relevance ?? '—'}</td>
                            <td class="score-cell ${scoreCellClass(r.context_recall)}">${r.context_recall ?? '—'}</td>
                            <td class="score-cell ${scoreCellClass(r.completeness)}">${r.completeness ?? '—'}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}


// =============================================================================
// UTILITIES
// =============================================================================

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function escapeAttr(text) {
    if (!text) return '';
    return text.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function extractFilename(path) {
    if (!path) return 'unknown';
    return path.split(/[/\\]/).pop();
}
