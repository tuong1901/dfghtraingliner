import os
import json
import html

def highlight_text(text, labels):
    """
    Sorts and filters labels to wrap corresponding spans in the text with HTML highlighting tags.
    Handles overlapping or invalid spans by keeping the longest or first encountered.
    """
    valid_labels = []
    for lbl in labels:
        if isinstance(lbl, list) and len(lbl) == 3:
            start, end, label_type = lbl
            if isinstance(start, int) and isinstance(end, int) and isinstance(label_type, str):
                if 0 <= start < end <= len(text):
                    valid_labels.append((start, end, label_type))
    
    # Sort by start index ascending, then by end index descending (longer span first)
    valid_labels.sort(key=lambda x: (x[0], -x[1]))
    
    # Resolve overlapping spans (greedy approach: keep non-overlapping)
    non_overlapping = []
    last_end = 0
    for start, end, label_type in valid_labels:
        if start >= last_end:
            non_overlapping.append((start, end, label_type))
            last_end = end
            
    # Construct highlighted HTML
    parts = []
    curr = 0
    for start, end, label_type in non_overlapping:
        # Text before entity
        parts.append(html.escape(text[curr:start]))
        # Labeled entity
        entity_text = html.escape(text[start:end])
        # HTML tag for entity
        cls_name = label_type.lower()
        parts.append(
            f'<span class="entity {cls_name}" data-label="{label_type}">'
            f'{entity_text}'
            f'<span class="label-tag">{label_type}</span>'
            f'</span>'
        )
        curr = end
    # Remaining text
    parts.append(html.escape(text[curr:]))
    return "".join(parts), non_overlapping

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "cleaned_dataset.json")
    html_output_path = os.path.join(base_dir, "visualize_labels.html")
    
    print(f"Loading data from {json_path}...")
    if not os.path.exists(json_path):
        # Fallback to parent dir or current working dir
        json_path = "data_xin_1000_dong_gold_backup.json"
        if not os.path.exists(json_path):
            print("Error: JSON file not found!")
            return
            
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} items. Processing labels...")
    
    # Pre-process items for Javascript database
    processed_items = []
    stats = {
        "total_docs": len(data),
        "total_entities": 0,
        "entity_counts": {},
        "level_counts": {}
    }
    
    for idx, item in enumerate(data):
        text = item.get("text", "")
        labels = item.get("label", [])
        level = item.get("level", "N/A")
        
        # Highlight and resolve overlaps
        highlighted_html, clean_labels = highlight_text(text, labels)
        
        # Count stats
        stats["level_counts"][level] = stats["level_counts"].get(level, 0) + 1
        
        entities_list = []
        for start, end, label_type in clean_labels:
            entity_val = text[start:end]
            entities_list.append({
                "text": entity_val,
                "label": label_type,
                "start": start,
                "end": end
            })
            stats["total_entities"] += 1
            stats["entity_counts"][label_type] = stats["entity_counts"].get(label_type, 0) + 1
            
        processed_items.append({
            "index": idx + 1,
            "level": level,
            "highlighted_html": highlighted_html,
            "entities": entities_list,
            "total_entities": len(entities_list),
            "text_length": len(text)
        })
        
    # Generate HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Job Description Label Visualizer</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-main: #0b0f19;
            --bg-card: #151d30;
            --bg-hover: #1e2942;
            --border-color: #24324f;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent-color: #3b82f6;
            
            /* Entity HSL Colors */
            --color-skill: #38bdf8;
            --bg-skill: rgba(56, 189, 248, 0.12);
            --border-skill: rgba(56, 189, 248, 0.35);
            
            --color-experience: #4ade80;
            --bg-experience: rgba(74, 222, 128, 0.12);
            --border-experience: rgba(74, 222, 128, 0.35);
            
            --color-major: #c084fc;
            --bg-major: rgba(192, 132, 252, 0.12);
            --border-major: rgba(192, 132, 252, 0.35);
            
            --color-other: #cbd5e1;
            --bg-other: rgba(203, 213, 225, 0.12);
            --border-other: rgba(203, 213, 225, 0.35);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-main);
            color: var(--text-primary);
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
            padding: 24px;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}

        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
        }}

        .logo-container h1 {{
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        }}

        .logo-container p {{
            font-size: 14px;
            color: var(--text-secondary);
        }}

        /* Dashboard Stats */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 12px;
        }}

        .stat-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 4px;
            transition: all 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-2px);
            border-color: var(--accent-color);
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
        }}

        .stat-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            font-weight: 600;
        }}

        .stat-value {{
            font-size: 24px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}

        .stat-card.skill-stat {{ border-left: 4px solid var(--color-skill); }}
        .stat-card.experience-stat {{ border-left: 4px solid var(--color-experience); }}
        .stat-card.major-stat {{ border-left: 4px solid var(--color-major); }}

        /* Main Workspace Layout */
        .workspace {{
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 24px;
            flex-grow: 1;
        }}

        /* Sidebar Filters */
        .sidebar {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            height: fit-content;
            position: sticky;
            top: 24px;
        }}

        .filter-section-title {{
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .search-box {{
            width: 100%;
            background-color: var(--bg-main);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px 12px;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }}

        .search-box:focus {{
            border-color: var(--accent-color);
        }}

        .checkbox-group {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .checkbox-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            font-size: 14px;
            color: var(--text-secondary);
            user-select: none;
            transition: color 0.2s;
        }}

        .checkbox-item:hover {{
            color: var(--text-primary);
        }}

        .checkbox-item input {{
            cursor: pointer;
        }}

        .level-badge-filter {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}

        .filter-badge {{
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            background-color: var(--bg-main);
            border: 1px solid var(--border-color);
            cursor: pointer;
            transition: all 0.2s;
            color: var(--text-secondary);
        }}

        .filter-badge:hover, .filter-badge.active {{
            background-color: var(--accent-color);
            border-color: var(--accent-color);
            color: #ffffff;
        }}

        /* Content Area */
        .content-area {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        .controls-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .results-count {{
            font-size: 14px;
            color: var(--text-secondary);
        }}

        .pagination {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .btn {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
            user-select: none;
        }}

        .btn:hover:not(:disabled) {{
            border-color: var(--accent-color);
            background-color: var(--bg-hover);
        }}

        .btn:disabled {{
            opacity: 0.4;
            cursor: not-allowed;
        }}

        .page-indicator {{
            font-size: 14px;
            font-weight: 500;
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
        }}

        /* Document Cards */
        .doc-list {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}

        .doc-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            transition: border-color 0.3s;
            position: relative;
        }}

        .doc-card:hover {{
            border-color: rgba(59, 130, 246, 0.4);
        }}

        .doc-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            padding-bottom: 12px;
        }}

        .doc-index {{
            font-size: 16px;
            font-weight: 600;
            color: var(--accent-color);
            font-family: 'JetBrains Mono', monospace;
        }}

        .doc-meta {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .level-badge {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.05em;
            background-color: rgba(245, 158, 11, 0.15);
            border: 1px solid rgba(245, 158, 11, 0.3);
            color: #fbbf24;
        }}

        .doc-text {{
            white-space: pre-wrap;
            font-size: 14px;
            line-height: 1.7;
            color: var(--text-primary);
            background-color: var(--bg-main);
            padding: 18px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            max-height: 400px;
            overflow-y: auto;
        }}

        /* Entity Highlighting Styles */
        .entity {{
            display: inline-flex;
            align-items: center;
            flex-wrap: wrap;
            border-radius: 4px;
            padding: 0 4px;
            margin: 0 2px;
            font-weight: 500;
            position: relative;
            cursor: pointer;
            transition: outline 0.1s ease;
        }}

        .entity:hover {{
            outline: 2px solid currentColor;
        }}

        .entity.skill {{
            background-color: var(--bg-skill);
            border: 1px solid var(--border-skill);
            color: var(--color-skill);
        }}

        .entity.experience {{
            background-color: var(--bg-experience);
            border: 1px solid var(--border-experience);
            color: var(--color-experience);
        }}

        .entity.major {{
            background-color: var(--bg-major);
            border: 1px solid var(--border-major);
            color: var(--color-major);
        }}

        .label-tag {{
            font-size: 9px;
            font-weight: 700;
            text-transform: uppercase;
            margin-left: 6px;
            opacity: 0.8;
            letter-spacing: 0.05em;
            vertical-align: middle;
            background-color: rgba(0, 0, 0, 0.2);
            padding: 1px 4px;
            border-radius: 3px;
        }}

        /* Footer Entity List inside Cards */
        .card-entities {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 4px;
        }}

        .entity-pill {{
            font-size: 12px;
            padding: 4px 8px;
            border-radius: 9999px;
            background-color: var(--bg-main);
            border: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .pill-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }}

        .pill-dot.skill {{ background-color: var(--color-skill); }}
        .pill-dot.experience {{ background-color: var(--color-experience); }}
        .pill-dot.major {{ background-color: var(--color-major); }}

        /* Scrollbars */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}

        ::-webkit-scrollbar-track {{
            background: var(--bg-main);
        }}

        ::-webkit-scrollbar-thumb {{
            background: var(--border-color);
            border-radius: 4px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: var(--text-secondary);
        }}

        /* Responsive Layout */
        @media (max-width: 900px) {{
            .workspace {{
                grid-template-columns: 1fr;
            }}
            .sidebar {{
                position: static;
            }}
        }}
    </style>
</head>
<body>

    <header>
        <div class="logo-container">
            <h1>Job Description Label Visualizer</h1>
            <p>Interactive tool to review gold labels (SKILL, EXPERIENCE, MAJOR) and level classification</p>
        </div>
    </header>

    <div class="stats-grid">
        <div class="stat-card">
            <span class="stat-label">Total Documents</span>
            <span class="stat-value">{stats["total_docs"]}</span>
        </div>
        <div class="stat-card">
            <span class="stat-label">Total Entities</span>
            <span class="stat-value">{stats["total_entities"]}</span>
        </div>
        <div class="stat-card skill-stat">
            <span class="stat-label">Skills</span>
            <span class="stat-value">{stats["entity_counts"].get("SKILL", 0)}</span>
        </div>
        <div class="stat-card experience-stat">
            <span class="stat-label">Experience</span>
            <span class="stat-value">{stats["entity_counts"].get("EXPERIENCE", 0)}</span>
        </div>
        <div class="stat-card major-stat">
            <span class="stat-label">Major</span>
            <span class="stat-value">{stats["entity_counts"].get("MAJOR", 0)}</span>
        </div>
    </div>

    <div class="workspace">
        <!-- Sidebar Filters -->
        <aside class="sidebar">
            <div>
                <span class="filter-section-title">Search Texts</span>
                <input type="text" id="text-search" class="search-box" placeholder="Type to search content...">
            </div>

            <div>
                <span class="filter-section-title">Entity Types</span>
                <div class="checkbox-group">
                    <label class="checkbox-item">
                        <input type="checkbox" id="check-skill" checked>
                        <span>Skills</span>
                    </label>
                    <label class="checkbox-item">
                        <input type="checkbox" id="check-exp" checked>
                        <span>Experience</span>
                    </label>
                    <label class="checkbox-item">
                        <input type="checkbox" id="check-major" checked>
                        <span>Major</span>
                    </label>
                </div>
            </div>

            <div>
                <span class="filter-section-title">Job Level</span>
                <div class="level-badge-filter" id="level-filter-container">
                    <span class="filter-badge active" data-level="ALL">ALL</span>
                    {" ".join([f'<span class="filter-badge" data-level="{lvl}">{lvl} ({cnt})</span>' for lvl, cnt in sorted(stats["level_counts"].items())])}
                </div>
            </div>
            
            <div>
                <span class="filter-section-title">Min Entities Count</span>
                <input type="number" id="min-entities" class="search-box" value="0" min="0" placeholder="Minimum entities...">
            </div>
        </aside>

        <!-- Content Area -->
        <main class="content-area">
            <div class="controls-row">
                <div class="results-count" id="results-summary">
                    Showing 0 - 0 of 0 documents
                </div>
                <div class="pagination">
                    <button class="btn" id="btn-prev" disabled>Previous</button>
                    <span class="page-indicator" id="page-indicator">Page 1 / 1</span>
                    <button class="btn" id="btn-next" disabled>Next</button>
                </div>
            </div>

            <div class="doc-list" id="document-list-container">
                <!-- Cards rendered dynamically -->
            </div>
        </main>
    </div>

    <script>
        // Injecting the processed database dynamically
        const dataset = {json.dumps(processed_items, ensure_ascii=False)};
        
        let filteredDataset = [...dataset];
        let currentPage = 1;
        const itemsPerPage = 10;
        let selectedLevel = "ALL";
        
        // Element references
        const searchInput = document.getElementById("text-search");
        const checkSkill = document.getElementById("check-skill");
        const checkExp = document.getElementById("check-exp");
        const checkMajor = document.getElementById("check-major");
        const minEntitiesInput = document.getElementById("min-entities");
        const levelBadges = document.querySelectorAll("#level-filter-container .filter-badge");
        
        const resultsSummary = document.getElementById("results-summary");
        const pageIndicator = document.getElementById("page-indicator");
        const btnPrev = document.getElementById("btn-prev");
        const btnNext = document.getElementById("btn-next");
        const docListContainer = document.getElementById("document-list-container");

        // Filter Logic
        function applyFilters() {{
            const searchVal = searchInput.value.toLowerCase().trim();
            const minEntities = parseInt(minEntitiesInput.value) || 0;
            const wantSkill = checkSkill.checked;
            const wantExp = checkExp.checked;
            const wantMajor = checkMajor.checked;
            
            filteredDataset = dataset.filter(item => {{
                // Level filter
                if (selectedLevel !== "ALL" && item.level !== selectedLevel) {{
                    return false;
                }}
                
                // Text search
                if (searchVal && !item.highlighted_html.toLowerCase().includes(searchVal)) {{
                    return false;
                }}
                
                // Min entities count
                if (item.total_entities < minEntities) {{
                    return false;
                }}
                
                // Entity types presence (if checked, must have at least one or we check item entities)
                // Filter the labels shown in HTML using CSS later or just filter dataset here.
                // For dataset level filter: if we search specifically, we can let CSS handle entity type visibility inside card,
                // but if someone searches for docs containing specific types, we can filter them here:
                if (!wantSkill || !wantExp || !wantMajor) {{
                    // check if doc has the active types
                    let hasMatch = false;
                    if (item.entities.length === 0) {{
                        // if we want nothing, matches empty docs too
                        return true; 
                    }}
                    for (const ent of item.entities) {{
                        if (ent.label === 'SKILL' && wantSkill) hasMatch = true;
                        if (ent.label === 'EXPERIENCE' && wantExp) hasMatch = true;
                        if (ent.label === 'MAJOR' && wantMajor) hasMatch = true;
                    }}
                    if (!hasMatch && item.entities.length > 0) return false;
                }}
                
                return true;
            }});
            
            currentPage = 1;
            renderPage();
            updateCSSVisibility();
        }}
        
        function updateCSSVisibility() {{
            // Use CSS variables or dynamic styles to show/hide entities
            const wantSkill = checkSkill.checked;
            const wantExp = checkExp.checked;
            const wantMajor = checkMajor.checked;
            
            let styleEl = document.getElementById("dynamic-visibility-styles");
            if (!styleEl) {{
                styleEl = document.createElement("style");
                styleEl.id = "dynamic-visibility-styles";
                document.head.appendChild(styleEl);
            }}
            
            let cssRules = "";
            if (!wantSkill) {{
                cssRules += `
                    .entity.skill {{
                        background: transparent !important;
                        border: none !important;
                        color: inherit !important;
                        padding: 0 !important;
                        margin: 0 !important;
                    }}
                    .entity.skill .label-tag {{ display: none !important; }}
                    .entity-pill.pill-skill {{ display: none !important; }}
                `;
            }}
            if (!wantExp) {{
                cssRules += `
                    .entity.experience {{
                        background: transparent !important;
                        border: none !important;
                        color: inherit !important;
                        padding: 0 !important;
                        margin: 0 !important;
                    }}
                    .entity.experience .label-tag {{ display: none !important; }}
                    .entity-pill.pill-experience {{ display: none !important; }}
                `;
            }}
            if (!wantMajor) {{
                cssRules += `
                    .entity.major {{
                        background: transparent !important;
                        border: none !important;
                        color: inherit !important;
                        padding: 0 !important;
                        margin: 0 !important;
                    }}
                    .entity.major .label-tag {{ display: none !important; }}
                    .entity-pill.pill-major {{ display: none !important; }}
                `;
            }}
            styleEl.innerHTML = cssRules;
        }}

        // Render Page
        function renderPage() {{
            docListContainer.innerHTML = "";
            
            const totalItems = filteredDataset.length;
            const totalPages = Math.ceil(totalItems / itemsPerPage) || 1;
            
            if (currentPage > totalPages) currentPage = totalPages;
            
            const startIndex = (currentPage - 1) * itemsPerPage;
            const endIndex = Math.min(startIndex + itemsPerPage, totalItems);
            
            // Update controls
            if (totalItems === 0) {{
                resultsSummary.innerText = "No documents match filters";
                pageIndicator.innerText = "Page 0 / 0";
                btnPrev.disabled = true;
                btnNext.disabled = true;
                docListContainer.innerHTML = `<div style="text-align: center; padding: 40px; color: var(--text-secondary); background: var(--bg-card); border: 1px dashed var(--border-color); border-radius: 12px;">Không tìm thấy kết quả nào khớp với bộ lọc.</div>`;
                return;
            }}
            
            resultsSummary.innerText = `Showing ${{startIndex + 1}} - ${{endIndex}} of ${{totalItems}} documents`;
            pageIndicator.innerText = `Page ${{currentPage}} / ${{totalPages}}`;
            
            btnPrev.disabled = currentPage === 1;
            btnNext.disabled = currentPage === totalPages;
            
            // Slice page data and render
            const pageData = filteredDataset.slice(startIndex, endIndex);
            pageData.forEach(item => {{
                // Generate pills summary
                let pillsHtml = "";
                item.entities.forEach(ent => {{
                    const lblLower = ent.label.toLowerCase();
                    pillsHtml += `
                        <div class="entity-pill pill-${{lblLower}}">
                            <span class="pill-dot ${{lblLower}}"></span>
                            <strong>${{ent.text}}</strong>
                            <span style="opacity: 0.6; font-size: 10px;">${{ent.label}}</span>
                        </div>
                    `;
                }});
                
                const card = document.createElement("div");
                card.className = "doc-card";
                card.innerHTML = `
                    <div class="doc-header">
                        <span class="doc-index">Document #${{item.index}}</span>
                        <div class="doc-meta">
                            <span class="level-badge">LEVEL: ${{item.level}}</span>
                            <span style="font-size: 12px; color: var(--text-secondary); font-family: 'JetBrains Mono';">${{item.text_length}} chars</span>
                        </div>
                    </div>
                    <div class="doc-text">${{item.highlighted_html}}</div>
                    <div class="card-entities">
                        ${{pillsHtml || '<span style="font-size:12px; color: var(--text-secondary); font-style:italic;">No entities annotated</span>'}}
                    </div>
                `;
                docListContainer.appendChild(card);
            }});
        }}

        // Listeners
        searchInput.addEventListener("input", applyFilters);
        checkSkill.addEventListener("change", applyFilters);
        checkExp.addEventListener("change", applyFilters);
        checkMajor.addEventListener("change", applyFilters);
        minEntitiesInput.addEventListener("input", applyFilters);
        
        levelBadges.forEach(badge => {{
            badge.addEventListener("click", () => {{
                levelBadges.forEach(b => b.classList.remove("active"));
                badge.classList.add("active");
                selectedLevel = badge.getAttribute("data-level");
                applyFilters();
            }});
        }});
        
        btnPrev.addEventListener("click", () => {{
            if (currentPage > 1) {{
                currentPage--;
                renderPage();
                window.scrollTo({{ top: 0, behavior: 'smooth' }});
            }}
        }});
        
        btnNext.addEventListener("click", () => {{
            const totalPages = Math.ceil(filteredDataset.length / itemsPerPage);
            if (currentPage < totalPages) {{
                currentPage++;
                renderPage();
                window.scrollTo({{ top: 0, behavior: 'smooth' }});
            }}
        }});
        
        // Initial setup
        applyFilters();
    </script>
</body>
</html>
"""
    
    print(f"Writing visualization HTML to {html_output_path}...")
    with open(html_output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print("Success! Visualization page generated successfully.")

if __name__ == "__main__":
    main()
