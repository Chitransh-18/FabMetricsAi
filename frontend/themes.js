/**
 * theme-gallery.js
 * ------------------------------------------------------------------
 * Self-contained Theme Gallery module for WaferScan AI.
 *
 * Owns everything about the theme picker: markup, preset data (55
 * presets across 4 sections), the swatch grid, and the toggleable
 * Manual Adjust panel.
 *
 * Talks to the rest of the app only through:
 *   - CSS custom properties on <html> (--primary-color, --secondary-color,
 *     --base-bg, --card-bg, --border-radius), which index.html already
 *     reads in its <style> block
 *   - window.openThemeModal(), called by the header button's onclick
 *
 * Nothing in index.html's core script (uploads, batch processing, tabs,
 * showroom, encyclopedia) touches this file or vice versa.
 * ------------------------------------------------------------------
 */
(function () {
    'use strict';

    // ---- Preset data (55 total) ---------------------------------------
    const SECTIONS = [
        {
            key: 'custom', label: 'Custom Presets',
            presets: [
                { name: 'Midnight Ocean', primary: '#22d3ee', secondary: '#3b82f6', bg: '#080c14', card: 'rgba(13,20,35,0.85)' },
                { name: 'Sunset Bliss', primary: '#f97316', secondary: '#ec4899', bg: '#1e1b4b', card: 'rgba(30,27,75,0.85)' },
                { name: 'Forest Whisper', primary: '#10b981', secondary: '#059669', bg: '#022c22', card: 'rgba(2,44,34,0.85)' },
                { name: 'Lavender Dream', primary: '#a855f7', secondary: '#6366f1', bg: '#0f172a', card: 'rgba(30,41,59,0.85)' },
                { name: 'Crimson Edge', primary: '#ef4444', secondary: '#f87171', bg: '#1a0505', card: 'rgba(40,10,10,0.85)' },
                { name: 'Peach Cream', primary: '#fb923c', secondary: '#fbbf24', bg: '#1c1006', card: 'rgba(50,30,10,0.85)' },
                { name: 'Indigo Storm', primary: '#6366f1', secondary: '#818cf8', bg: '#0b0d2e', card: 'rgba(20,22,60,0.85)' },
                { name: 'Mint Clarity', primary: '#2dd4bf', secondary: '#5eead4', bg: '#041e1c', card: 'rgba(6,40,38,0.85)' },
                { name: 'Obsidian Night', primary: '#a78bfa', secondary: '#c4b5fd', bg: '#05050a', card: 'rgba(15,15,25,0.85)' },
                { name: 'Golden Hour', primary: '#f59e0b', secondary: '#fbbf24', bg: '#1a1205', card: 'rgba(40,28,10,0.85)' }
            ]
        },
        {
            key: 'company', label: 'Company Themes',
            presets: [
                { name: 'AWS Orange', primary: '#FF9900', secondary: '#FF9900', bg: '#0d1b2a', card: '#16232f' },
                { name: 'Microsoft Blue', primary: '#0078D4', secondary: '#0078D4', bg: '#0b1a2b', card: '#12233a' },
                { name: 'GitHub Dark', primary: '#58a6ff', secondary: '#58a6ff', bg: '#0d1117', card: '#161b22' },
                { name: 'VS Code', primary: '#007ACC', secondary: '#3b82f6', bg: '#1e1e1e', card: '#252526' },
                { name: 'Postman Orange', primary: '#FF6C37', secondary: '#FF6C37', bg: '#1a1005', card: '#241608' },
                { name: 'Render Blue', primary: '#46E3B7', secondary: '#46E3B7', bg: '#0b0e14', card: '#151a24' },
                { name: 'Slack Variant', primary: '#4A154B', secondary: '#ECB22E', bg: '#0f0a10', card: '#1a1319' },
                { name: 'Figma Design', primary: '#F24E1E', secondary: '#A259FF', bg: '#100a1a', card: '#1a1225' },
                { name: 'Vercel Dark', primary: '#ffffff', secondary: '#a3a3a3', bg: '#000000', card: '#111111' },
                { name: 'Stripe Modern', primary: '#635BFF', secondary: '#00D4FF', bg: '#0a0a1a', card: '#14142a' }
            ]
        },
        {
            key: 'modern', label: 'Modern Presets',
            presets: [
                { name: 'Dribbble Pink', primary: '#EA4C89', secondary: '#EA4C89', bg: '#fdf2f8', card: '#ffffff' },
                { name: 'Twitch Purple', primary: '#9146FF', secondary: '#9146FF', bg: '#0e0e10', card: '#18181b' },
                { name: 'Dropbox Blue', primary: '#0061FF', secondary: '#0061FF', bg: '#f7f9fc', card: '#ffffff' },
                { name: 'Coral Sunset', primary: '#ff6b6b', secondary: '#ffa07a', bg: '#1a0e0e', card: '#2a1515' },
                { name: 'Ocean Breeze', primary: '#38bdf8', secondary: '#0ea5e9', bg: '#041826', card: '#0a2436' },
                { name: 'Rose Gold', primary: '#f5a3c7', secondary: '#f8c471', bg: '#1a1214', card: '#261a1d' },
                { name: 'Emerald City', primary: '#34d399', secondary: '#10b981', bg: '#04140f', card: '#0a241a' },
                { name: 'Cyber Punk', primary: '#f472b6', secondary: '#22d3ee', bg: '#0a0014', card: '#150524' },
                { name: 'Arctic Frost', primary: '#93c5fd', secondary: '#bfdbfe', bg: '#0a1420', card: '#142438' },
                { name: 'Autumn Leaves', primary: '#ea580c', secondary: '#ca8a04', bg: '#1a0f05', card: '#2a1a0a' }
            ]
        },
        {
            key: 'additional', label: 'Additional Themes',
            presets: [
                { name: 'Slack Dark', primary: '#E01E5A', secondary: '#36C5F0', bg: '#1a1d21', card: '#222529' },
                { name: 'Linear App', primary: '#5E6AD2', secondary: '#8B92F0', bg: '#0d0e14', card: '#17181f' },
                { name: 'Firebase Orange', primary: '#FFA000', secondary: '#FFCA28', bg: '#1a1502', card: '#262008' },
                { name: 'Docker Blue', primary: '#2496ED', secondary: '#2496ED', bg: '#061826', card: '#0d2438' },
                { name: 'GitHub Light', primary: '#0969DA', secondary: '#1F883D', bg: '#ffffff', card: '#f6f8fa' },
                { name: 'Slack Pink', primary: '#E01E5A', secondary: '#ECB22E', bg: '#1c1013', card: '#29181c' },
                { name: 'Heroku Purple', primary: '#6762A6', secondary: '#C9CBFF', bg: '#120f1e', card: '#1e1a30' },
                { name: 'MongoDB Green', primary: '#00ED64', secondary: '#00684A', bg: '#001217', card: '#001e2b' },
                { name: 'Figma Pink', primary: '#FF7262', secondary: '#A259FF', bg: '#1a0f10', card: '#26171a' },
                { name: 'Notion Dark', primary: '#ffffff', secondary: '#9b9a97', bg: '#191919', card: '#202020' },
                { name: 'Discord Dark', primary: '#5865F2', secondary: '#5865F2', bg: '#202225', card: '#2f3136' },
                { name: 'LinkedIn Blue', primary: '#0A66C2', secondary: '#0A66C2', bg: '#0a1520', card: '#12202e' },
                { name: 'Telegram Blue', primary: '#26A5E4', secondary: '#26A5E4', bg: '#0e1621', card: '#17212b' },
                { name: 'GitLab Orange', primary: '#FC6D26', secondary: '#E24329', bg: '#1a0f08', card: '#26170e' },
                { name: 'Trello Blue', primary: '#0079BF', secondary: '#00C2E0', bg: '#08131a', card: '#0f2029' },
                { name: 'Slack Aqua', primary: '#1FA2AC', secondary: '#2EB67D', bg: '#051615', card: '#0c2422' },
                { name: 'Asana Blue', primary: '#273347', secondary: '#F06A6A', bg: '#0a0e15', card: '#131a25' },
                { name: 'Monday.com Blue', primary: '#0073EA', secondary: '#00C875', bg: '#071420', card: '#0f2031' },
                { name: 'Notion Purple', primary: '#9B87F5', secondary: '#E4DFFD', bg: '#120f1e', card: '#1c1830' },
                { name: 'Canvas Green', primary: '#00AC18', secondary: '#E8F0E9', bg: '#041a06', card: '#0a2810' },
                { name: 'Reddit Orange', primary: '#FF4500', secondary: '#FF8717', bg: '#1a0800', card: '#261200' },
                { name: 'Twitter Blue', primary: '#1D9BF0', secondary: '#1D9BF0', bg: '#05131e', card: '#0d1f2e' },
                { name: 'Spotify Green', primary: '#1DB954', secondary: '#1ED760', bg: '#050f08', card: '#0a1a0e' },
                { name: 'Pinterest Red', primary: '#E60023', secondary: '#E60023', bg: '#1a0508', card: '#260a0e' },
                { name: 'WhatsApp Green', primary: '#25D366', secondary: '#128C7E', bg: '#05130a', card: '#0c1f14' }
            ]
        }
    ];

    const TOTAL_PRESETS = SECTIONS.reduce((sum, s) => sum + s.presets.length, 0);

    // ---- Markup builders ------------------------------------------------
    function swatchBarHtml(preset) {
        return `
            <div class="theme-swatch-bar" style="background: linear-gradient(90deg, ${preset.bg} 0%, ${preset.bg} 38%, ${preset.card} 38%, ${preset.card} 68%, ${preset.primary} 68%, ${preset.primary} 100%);">
                <svg class="theme-swatch-icon" viewBox="0 0 24 24" fill="currentColor"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>
            </div>`;
    }

    function presetButtonHtml(preset) {
        return `
            <button type="button" data-theme-primary="${preset.primary}" data-theme-secondary="${preset.secondary}" data-theme-bg="${preset.bg}" data-theme-card="${preset.card}"
                class="theme-preset-btn text-left group">
                ${swatchBarHtml(preset)}
                <span class="text-xs font-bold block text-slate-200 mt-2 group-hover:text-cyan-300 transition-colors">${preset.name}</span>
            </button>`;
    }

    function sectionHtml(section) {
        return `
            <div class="theme-section" data-section="${section.key}">
                <div class="flex items-center gap-2 mb-3">
                    <svg class="w-3.5 h-3.5 text-slate-500" viewBox="0 0 24 24" fill="currentColor"><path d="M4 4h7v7H4V4zm9 0h7v7h-7V4zM4 13h7v7H4v-7zm9 0h7v7h-7v-7z"/></svg>
                    <span class="text-[10px] uppercase font-bold text-slate-500 tracking-widest">${section.label}</span>
                    <div class="flex-grow h-px bg-white/5"></div>
                </div>
                <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
                    ${section.presets.map(presetButtonHtml).join('')}
                </div>
            </div>`;
    }

    function colorFieldHtml(id, label, value) {
        return `
            <div>
                <label class="text-[11px] font-bold text-slate-400 block mb-2 tracking-wide">${label}</label>
                <div class="relative flex items-center bg-black border border-white/10 rounded-lg overflow-hidden h-10">
                    <div class="w-8 h-8 ml-1 rounded shrink-0" style="background:${value}" id="${id}-swatch"></div>
                    <span class="ml-2 text-xs font-mono text-slate-300" id="${id}-hex">${value.toUpperCase()}</span>
                    <input type="color" id="${id}" value="${value.length === 7 ? value : '#22d3ee'}" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer">
                </div>
            </div>`;
    }

    function buildModalMarkup() {
        return `
        <div id="theme-modal" class="hidden fixed inset-0 bg-black/70 backdrop-blur-md z-50 flex items-center justify-center p-4">
            <div class="w-full max-w-5xl bg-zinc-950 rounded-2xl border border-white/10 overflow-hidden shadow-2xl flex flex-col max-h-[88vh]">
                <div class="p-4 border-b border-white/5 flex justify-between items-center bg-black/40">
                    <div class="flex items-center space-x-2">
                        <svg class="w-5 h-5 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                        <h3 class="text-sm font-bold uppercase tracking-wider text-slate-200">Theme Gallery</h3>
                    </div>
                    <button id="theme-manual-toggle" type="button" class="px-3 py-1.5 rounded-full border border-white/30 text-[11px] font-bold uppercase tracking-wide text-slate-200 flex items-center gap-1.5 hover:border-white/60 transition-colors">
                        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 21v-7m0-4V3m8 18v-9m0-4V3m8 18v-5m0-4V3M1 14h6M9 8h6M17 16h6"></path></svg>
                        <span>Manual Adjust</span>
                        <span id="theme-manual-toggle-x" class="hidden ml-0.5">&times;</span>
                    </button>
                </div>

                <div class="p-6 overflow-y-auto flex-grow">
                    <div id="theme-gallery-view">
                        ${SECTIONS.map(sectionHtml).join('')}
                    </div>

                    <div id="theme-manual-view" class="hidden">
                        <div class="grid grid-cols-1 sm:grid-cols-4 gap-4">
                            ${colorFieldHtml('input-color-primary', 'PRIMARY', '#22d3ee')}
                            ${colorFieldHtml('input-color-secondary', 'SECONDARY', '#3b82f6')}
                            ${colorFieldHtml('input-color-bg', 'BASE BG', '#080c14')}
                            ${colorFieldHtml('input-color-card', 'CARD BG', '#0d1423')}
                        </div>
                        <div class="mt-6">
                            <div class="flex justify-between text-[11px] font-bold text-slate-400 mb-1">
                                <label>RADIUS</label>
                                <span id="radius-val-display">12PX</span>
                            </div>
                            <input type="range" id="input-radius-slider" min="0" max="24" value="12" class="w-full accent-cyan-400">
                        </div>
                    </div>
                </div>

                <div class="p-4 bg-black/40 border-t border-white/5 flex justify-between items-center">
                    <span class="text-[10px] font-mono text-slate-500 uppercase tracking-wide">${TOTAL_PRESETS} presets loaded</span>
                    <div class="flex space-x-2">
                        <button id="theme-modal-discard-btn" type="button" class="bg-zinc-800 hover:bg-zinc-700 text-slate-200 text-xs font-bold px-4 py-2 rounded-lg transition-colors">Discard</button>
                        <button id="theme-modal-confirm-btn" type="button" class="bg-white hover:bg-slate-200 text-slate-950 text-xs font-bold px-4 py-2 rounded-lg transition-colors">Confirm Selection</button>
                    </div>
                </div>
            </div>
        </div>`;
    }

    // ---- Inline styles the swatch bars/icons need (kept minimal; everything else is Tailwind) ----
    function injectStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .theme-swatch-bar { position: relative; height: 3rem; border-radius: 0.5rem; overflow: hidden; border: 1px solid rgba(255,255,255,0.08); }
            .theme-swatch-icon { position: absolute; top: 4px; right: 4px; width: 12px; height: 12px; color: rgba(255,255,255,0.55); filter: drop-shadow(0 0 2px rgba(0,0,0,0.6)); }
            .theme-preset-btn { transition: transform 0.15s ease; }
            .theme-preset-btn:hover { transform: translateY(-2px); }
        `;
        document.head.appendChild(style);
    }

    // ---- Core theme logic -----------------------------------------------
    function applyTheme(primary, secondary, bg, card) {
        document.documentElement.style.setProperty('--primary-color', primary);
        document.documentElement.style.setProperty('--secondary-color', secondary);
        document.documentElement.style.setProperty('--base-bg', bg);
        document.documentElement.style.setProperty('--card-bg', card);
    }

    function syncColorField(id, value) {
        document.getElementById(`${id}-swatch`).style.background = value;
        document.getElementById(`${id}-hex`).innerText = value.toUpperCase();
    }

    function manualAdjustTheme() {
        const p = document.getElementById('input-color-primary').value;
        const s = document.getElementById('input-color-secondary').value;
        const b = document.getElementById('input-color-bg').value;
        const c = document.getElementById('input-color-card').value;
        applyTheme(p, s, b, c);
    }

    function manualAdjustRadius() {
        const val = document.getElementById('input-radius-slider').value;
        document.getElementById('radius-val-display').innerText = `${val}PX`;
        document.documentElement.style.setProperty('--border-radius', `${val}px`);
    }

    function setManualMode(active) {
        document.getElementById('theme-gallery-view').classList.toggle('hidden', active);
        document.getElementById('theme-manual-view').classList.toggle('hidden', !active);
        document.getElementById('theme-manual-toggle-x').classList.toggle('hidden', !active);
        document.getElementById('theme-manual-toggle').classList.toggle('border-cyan-400', active);
        document.getElementById('theme-manual-toggle').classList.toggle('text-cyan-300', active);
    }

    function openThemeModal() {
        const modal = document.getElementById('theme-modal');
        if (modal) modal.classList.remove('hidden');
    }

    function closeThemeModal() {
        const modal = document.getElementById('theme-modal');
        if (modal) modal.classList.add('hidden');
    }

    // ---- Wiring -----------------------------------------------------------
    function initThemeGallery() {
        injectStyles();

        const wrapper = document.createElement('div');
        wrapper.innerHTML = buildModalMarkup();
        document.body.appendChild(wrapper.firstElementChild);

        const modal = document.getElementById('theme-modal');

        // Preset selection (event delegation)
        modal.addEventListener('click', (e) => {
            const presetBtn = e.target.closest('.theme-preset-btn');
            if (presetBtn) {
                applyTheme(
                    presetBtn.dataset.themePrimary,
                    presetBtn.dataset.themeSecondary,
                    presetBtn.dataset.themeBg,
                    presetBtn.dataset.themeCard
                );
            }
        });

        // Manual Adjust toggle
        let manualActive = false;
        document.getElementById('theme-manual-toggle').addEventListener('click', () => {
            manualActive = !manualActive;
            setManualMode(manualActive);
        });

        // Manual color/radius controls
        ['input-color-primary', 'input-color-secondary', 'input-color-bg', 'input-color-card'].forEach((id) => {
            document.getElementById(id).addEventListener('input', (e) => {
                syncColorField(id, e.target.value);
                manualAdjustTheme();
            });
        });
        document.getElementById('input-radius-slider').addEventListener('input', manualAdjustRadius);

        // Discard / confirm / click-outside close
        document.getElementById('theme-modal-discard-btn').addEventListener('click', closeThemeModal);
        document.getElementById('theme-modal-confirm-btn').addEventListener('click', closeThemeModal);
        modal.addEventListener('click', (e) => {
            if (e.target.id === 'theme-modal') closeThemeModal();
        });
    }

    // Expose only what the header button needs (it calls openThemeModal() via onclick)
    window.openThemeModal = openThemeModal;
    window.closeThemeModal = closeThemeModal;

    document.addEventListener('DOMContentLoaded', initThemeGallery);
})();
