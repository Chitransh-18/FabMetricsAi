/**
 * Cleanroom AI Assistant Component Module
 * Powered by Google Gemini API with Domain Strictness Safeguards
 */

(function () {
    'use strict';

    // 🔑 Insert your Gemini API Key here
    const GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE";

    // Domain System Prompt to enforce strict focus & accuracy
    const SYSTEM_INSTRUCTION = `
You are the "Cleanroom AI Tutor", an expert AI assistant embedded inside the WaferScan AI industrial platform.
YOUR PURPOSE:
- Answer engineering, semiconductor manufacturing, cleanroom protocol, wafer defect classification (Scratch, Donut, Edge-Ring, Loc, Center, etc.), computer vision (OpenCV), and deep learning (ResNet-34) questions accurately and concisely.
- Keep answers professional, concise, and structured (use bolding or short bullet points where helpful).

STRICT GUARDRAILS:
- If the user asks a question completely unrelated to semiconductor fabrication, cleanrooms, computer vision, machine learning, physics, material science, or this software project (e.g., asking about cooking, pop culture, sports, general politics), REJECT IT POLITELY.
- Example Rejection: "I am specialized solely in cleanroom operations, semiconductor yield analysis, and wafer defect detection. Please ask a question related to these domains!"
`;

    // 1. Inject Custom CSS
    const style = document.createElement('style');
    style.innerHTML = `
        .chat-feed-scroll::-webkit-scrollbar { width: 4px; }
        .chat-feed-scroll::-webkit-scrollbar-thumb { background: rgba(56, 189, 248, 0.2); border-radius: 4px; }
    `;
    document.head.appendChild(style);

    // 2. Render UI into container on DOM ready
    window.addEventListener('DOMContentLoaded', () => {
        const chatContainer = document.getElementById('ai-tutor-container');
        if (!chatContainer) return;

        chatContainer.innerHTML = `
            <div class="glass-panel p-5 rounded-2xl border border-white/10 flex flex-col h-[400px] shadow-xl">
                <div class="flex items-center justify-between border-b border-slate-800 pb-3 mb-3">
                    <div class="flex items-center space-x-2">
                        <span class="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
                        <h4 class="text-sm font-bold tracking-wider text-slate-200 uppercase">Cleanroom AI Tutor</h4>
                    </div>
                    <span class="text-[10px] bg-cyan-950 text-cyan-400 px-2.5 py-0.5 rounded-full border border-cyan-800/50 font-mono">Gemini 2.5 Flash Active</span>
                </div>

                <!-- Chat Feed -->
                <div id="chat-feed" class="chat-feed-scroll flex-grow overflow-y-auto space-y-3 pr-2 text-sm font-mono text-slate-300">
                    <div class="bg-slate-900/80 border border-slate-800 p-3 rounded-xl max-w-[85%] shadow-sm">
                        <span class="text-[10px] text-cyan-400 font-bold block mb-1">SYSTEM TUTOR</span>
                        Hello Engineer! Ask me anything about semiconductor defect physics, ResNet-34 metrics, or cleanroom yield optimization.
                    </div>
                </div>

                <!-- Chat Input Area -->
                <div class="flex items-center space-x-2 mt-3 pt-2 border-t border-slate-800">
                    <input type="text" id="chat-input" placeholder="Ask about wafer defects, ResNet architecture, or CMP..." 
                           class="flex-grow bg-slate-950 border border-slate-800 rounded-xl px-4 py-2 text-sm text-white focus:outline-none focus:border-cyan-500 font-mono transition-colors">
                    <button id="chat-send-btn" class="bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-bold px-4 py-2 rounded-xl text-xs uppercase tracking-wider hover:opacity-90 transition active:scale-95 flex items-center space-x-1">
                        <span>Send</span>
                    </button>
                </div>
            </div>
        `;

        const inputEl = document.getElementById('chat-input');
        const sendBtn = document.getElementById('chat-send-btn');

        if (sendBtn && inputEl) {
            sendBtn.addEventListener('click', handleUserChat);
            inputEl.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') handleUserChat();
            });
        }
    });

    // 3. User Chat Handler
    async function handleUserChat() {
        const inputEl = document.getElementById('chat-input');
        const feed = document.getElementById('chat-feed');
        if (!inputEl || !feed) return;

        const query = inputEl.value.trim();
        if (!query) return;

        // Append User Message
        feed.innerHTML += `
            <div class="bg-cyan-950/40 border border-cyan-800/50 p-3 rounded-xl max-w-[85%] ml-auto text-right shadow-sm">
                <span class="text-[10px] text-slate-400 font-bold block mb-1">YOU</span>
                ${escapeHtml(query)}
            </div>
        `;
        inputEl.value = '';
        feed.scrollTop = feed.scrollHeight;

        // Typing Indicator Placeholder
        const loadingId = 'loading-' + Date.now();
        feed.innerHTML += `
            <div id="${loadingId}" class="bg-slate-900/80 border border-slate-800 p-3 rounded-xl max-w-[85%] shadow-sm">
                <span class="text-[10px] text-cyan-400 font-bold block mb-1">SYSTEM TUTOR</span>
                <span class="animate-pulse text-xs text-slate-400">Analyzing domain query...</span>
            </div>
        `;
        feed.scrollTop = feed.scrollHeight;

        // Call Gemini API
        try {
            const aiResponse = await callGeminiAPI(query);
            document.getElementById(loadingId)?.remove();

            feed.innerHTML += `
                <div class="bg-slate-900/80 border border-slate-800 p-3 rounded-xl max-w-[85%] shadow-sm">
                    <span class="text-[10px] text-cyan-400 font-bold block mb-1">SYSTEM TUTOR</span>
                    ${formatResponse(aiResponse)}
                </div>
            `;
        } catch (err) {
            document.getElementById(loadingId)?.remove();
            feed.innerHTML += `
                <div class="bg-red-950/30 border border-red-800/50 p-3 rounded-xl max-w-[85%] shadow-sm text-red-300">
                    <span class="text-[10px] text-red-400 font-bold block mb-1">SYSTEM ERROR</span>
                    Unable to contact Gemini API. Please check your API key or network connection.
                </div>
            `;
        }
        feed.scrollTop = feed.scrollHeight;
    }

    // 4. Direct Fetch Call to Gemini REST API with Intelligent Domain Fallback
    async function callGeminiAPI(prompt) {
        if (!GEMINI_API_KEY || GEMINI_API_KEY === "YOUR_GEMINI_API_KEY_HERE") {
            const p = prompt.toLowerCase();
            if (p.includes("scratch")) {
                return "**Scratch Defects** are linear physical scratches caused by mechanical pick-and-place grippers or transport slot track friction during wafer transfers across cleanroom rails.";
            } else if (p.includes("donut")) {
                return "**Donut Defects** present as concentric loops inside the interior wafer area, usually caused by chemical vapor deposition (CVD) gas distribution non-uniformity.";
            } else if (p.includes("edge") || p.includes("ring")) {
                return "**Edge-Ring Defects** manifest along the extreme perimeter of the wafer disk, typically caused by plasma etching edge-effect non-uniformities or clamp ring mechanical stress.";
            } else if (p.includes("resnet") || p.includes("f1") || p.includes("accuracy") || p.includes("model")) {
                return "**ResNet-34 Architecture**: Utilizes residual identity skip connections to prevent gradient degradation. Reaches **87.51% Macro-F1** on the WM-811K benchmark, outperforming HOG+RF (41.89%) and Shallow CNNs (43.13%).";
            } else if (p.includes("yield") || p.includes("cleanroom")) {
                return "**Cleanroom Operations**: ISO Class 1 to Class 5 cleanrooms enforce positive pressure, HEPA filtration, and strict gowning to prevent particulate contamination from destroying microchip gate geometry.";
            } else {
                return "**Cleanroom AI Knowledge Hub**: I can provide insights into **Scratch**, **Donut**, **Edge-Ring**, **Loc**, and **Random** defect physics, **ResNet-34** architecture metrics (87.51% Macro-F1), or **cleanroom yield protocols**. *(Optional: Set your Gemini API key in `frontend/chatbot.js` for live generative responses!)*";
            }
        }

        const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${GEMINI_API_KEY}`;

        const requestBody = {
            contents: [
                {
                    role: "user",
                    parts: [
                        { text: SYSTEM_INSTRUCTION },
                        { text: `User Question: ${prompt}` }
                    ]
                }
            ]
        };

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`API returned status ${response.status}`);
        }

        const data = await response.json();
        return data.candidates?.[0]?.content?.parts?.[0]?.text || "No response generated.";
    }

    // Basic markdown-to-HTML parser for bullet points & bold text
    function formatResponse(text) {
        let clean = escapeHtml(text);
        clean = clean.replace(/\*\*(.*?)\*\*/g, '<strong class="text-white">$1</strong>');
        clean = clean.replace(/\*(.*?)\*/g, '<em>$1</em>');
        clean = clean.replace(/\n/g, '<br>');
        return clean;
    }

    function escapeHtml(text) {
        return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }
})();