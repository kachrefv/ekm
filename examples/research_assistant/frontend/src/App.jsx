import React, { useState, useEffect, useRef } from 'react';
import {
    MessageSquare,
    Database,
    Moon,
    Send,
    Plus,
    Settings,
    Cpu,
    Activity,
    ChevronRight,
    Upload,
    RefreshCw,
    Sparkles,
    Box,
    Trash2,
    Layers
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { motion, AnimatePresence } from 'framer-motion';
import MeshVisualization from './MeshVisualization';

const API_BASE = 'http://127.0.0.1:8000';

function App() {
    const [workspaces, setWorkspaces] = useState([]);
    const [selectedWorkspace, setSelectedWorkspace] = useState('');
    const [sessions, setSessions] = useState([]);
    const [selectedSession, setSelectedSession] = useState('');
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [newWorkspaceName, setNewWorkspaceName] = useState('');
    const [showWorkspaceModal, setShowWorkspaceModal] = useState(false);
    const [viewMode, setViewMode] = useState('chat');
    const [persona, setPersona] = useState({ name: 'EKM Agent' });
    const [consciousness, setConsciousness] = useState(null);
    const [isBackendReady, setIsBackendReady] = useState(false);
    const [connectionError, setConnectionError] = useState(null);
    const [settings, setSettings] = useState({});
    const [showSettingsModal, setShowSettingsModal] = useState(false);
    const [includeChainOfThoughts, setIncludeChainOfThoughts] = useState(false);
    const [useAgenticSystem, setUseAgenticSystem] = useState(false);
    const [researchLoading, setResearchLoading] = useState(false);
    const [activeTasks, setActiveTasks] = useState([]);
    const [showResearchModal, setShowResearchModal] = useState(false);
    const [researchQuery, setResearchQuery] = useState('');

    const chatEndRef = useRef(null);
    const fileInputRef = useRef(null);
    const textareaRef = useRef(null);
    const iterationsRef = useRef(null);
    const formatRef = useRef(null);
    const includeChainOfThoughtsRef = useRef(null);
    const useAgenticSystemRef = useRef(null);

    // Initial Fetch & Connection Monitoring
    useEffect(() => {
        let pollInterval;

        const checkConnection = async () => {
            try {
                const res = await fetch(`${API_BASE}/workspaces`);
                if (res.ok) {
                    setIsBackendReady(true);
                    setConnectionError(null);
                    fetchWorkspaces();
                    if (pollInterval) clearInterval(pollInterval);
                }
            } catch (err) {
                console.log("Backend not ready yet...", err);
                setConnectionError("Waiting for EKM Sidecar to start...");
            }
        };

        checkConnection();
        pollInterval = setInterval(checkConnection, 3000);

        return () => clearInterval(pollInterval);
    }, []);

    // Fetch Sessions when Workspace changes
    useEffect(() => {
        if (selectedWorkspace) {
            fetchSessions(selectedWorkspace);
            fetchPersona(selectedWorkspace);
            fetchConsciousness(selectedWorkspace);
            fetchSettings(selectedWorkspace);
            fetchWorkspaceTasks(selectedWorkspace);
        } else {
            setSessions([]);
            setSelectedSession('');
            setMessages([]);
            setPersona({ name: 'EKM Agent' });
            setConsciousness(null);
            setActiveTasks([]);
        }
    }, [selectedWorkspace]);

    useEffect(() => {
        if (selectedSession) {
            fetchMessages(selectedSession);
        } else {
            setMessages([]);
        }
    }, [selectedSession]);

    // Real-time Task Updates (SSE)
    useEffect(() => {
        if (!selectedWorkspace || !isBackendReady) return;

        console.log("Connecting to task events for workspace:", selectedWorkspace);
        const eventSource = new EventSource(`${API_BASE}/tasks/events?workspace_id=${selectedWorkspace}`);

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (Array.isArray(data)) {
                    setActiveTasks(data);
                }
            } catch (err) {
                console.error("SSE Parse Error:", err);
            }
        };

        eventSource.onerror = (err) => {
            // console.error("SSE Error:", err);
            eventSource.close();
        };

        return () => {
            eventSource.close();
        };
    }, [selectedWorkspace, isBackendReady]);

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, loading]);

    const fetchWorkspaces = async () => {
        try {
            const res = await fetch(`${API_BASE}/workspaces`);
            const data = await res.json();
            setWorkspaces(data);
            if (data.length > 0 && !selectedWorkspace) {
                setSelectedWorkspace(data[0].id);
            }
        } catch (err) {
            console.error("Failed to fetch workspaces", err);
        }
    };

    const fetchSessions = async (workspaceId) => {
        try {
            const res = await fetch(`${API_BASE}/workspaces/${workspaceId}/sessions`);
            const data = await res.json();
            setSessions(data);
            if (data.length > 0) {
                setSelectedSession(data[0].id);
            } else {
                setSelectedSession('');
                setMessages([]);
            }
        } catch (err) {
            console.error("Failed to fetch sessions", err);
        }
    };

    const fetchMessages = async (sessionId) => {
        try {
            const res = await fetch(`${API_BASE}/sessions/${sessionId}/messages`);
            const data = await res.json();
            setMessages(data);
        } catch (err) {
            console.error("Failed to fetch messages", err);
        }
    };

    const fetchPersona = async (workspaceId) => {
        try {
            const res = await fetch(`${API_BASE}/workspaces/${workspaceId}/persona`);
            const data = await res.json();
            setPersona(data);
        } catch (err) { console.error("Persona fetch error", err); }
    };

    const fetchConsciousness = async (workspaceId) => {
        try {
            const res = await fetch(`${API_BASE}/workspaces/${workspaceId}/consciousness`);
            const data = await res.json();
            setConsciousness(data);
        } catch (err) { console.error("Consciousness fetch error", err); }
    };

    const fetchSettings = async (workspaceId) => {
        try {
            const res = await fetch(`${API_BASE}/workspaces/${workspaceId}/settings`);
            const data = await res.json();
            setSettings(data.settings);
        } catch (err) { console.error("Settings fetch error", err); }
    };

    const updateSettings = async () => {
        if (!selectedWorkspace) return;
        try {
            setLoading(true);
            const res = await fetch(`${API_BASE}/workspaces/${selectedWorkspace}/settings`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            });
            if (res.ok) alert("Workspace configuration updated.");
        } catch (err) { console.error("Settings update error", err); }
        finally { setLoading(false); setShowSettingsModal(false); }
    };

    const createWorkspace = async () => {
        if (!newWorkspaceName.trim()) return;
        try {
            const res = await fetch(`${API_BASE}/workspaces?name=${encodeURIComponent(newWorkspaceName)}`, { method: 'POST' });
            const data = await res.json();
            setWorkspaces(prev => [...prev, data]);
            setSelectedWorkspace(data.id);
            setNewWorkspaceName('');
            setShowWorkspaceModal(false);
        } catch (err) {
            console.error("Failed to create workspace", err);
        }
    };

    const createSession = async () => {
        if (!selectedWorkspace) return;
        try {
            const res = await fetch(`${API_BASE}/sessions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ workspace_id: selectedWorkspace, name: "New Chat" })
            });
            const data = await res.json();
            setSessions(prev => [data, ...prev]);
            setSelectedSession(data.id);
            setViewMode('chat');
        } catch (err) {
            console.error("Failed to create session", err);
        }
    };

    const deleteSession = async (e, sessionId) => {
        e.stopPropagation();
        if (!confirm("Delete this chat session?")) return;
        try {
            await fetch(`${API_BASE}/sessions/${sessionId}`, { method: 'DELETE' });
            setSessions(prev => prev.filter(s => s.id !== sessionId));
            if (selectedSession === sessionId) {
                setSelectedSession('');
                setMessages([]);
            }
        } catch (err) {
            console.error("Failed to delete session", err);
        }
    };

    const handleFileUpload = async (e) => {
        const files = e.target.files;
        if (!files || files.length === 0 || !selectedWorkspace) return;
        setLoading(true);
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) formData.append('files', files[i]);
        try {
            const res = await fetch(`${API_BASE}/train/${selectedWorkspace}`, { method: 'POST', body: formData });
            const data = await res.json();
            alert(`Training started on ${data.file_count} file(s).`);
        } catch (err) { console.error("Upload failure", err); }
        finally { setLoading(false); }
    };

    const sendMessage = async () => {
        if (!input.trim() || !selectedWorkspace) return;

        // Optimistic update
        const userMsg = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        const currentInput = input;
        setInput('');
        setLoading(true);

        try {
            const res = await fetch(`${API_BASE}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    workspace_id: selectedWorkspace,
                    session_id: selectedSession || null,
                    message: currentInput,
                    include_chain_of_thoughts: includeChainOfThoughts,
                    use_agentic_system: useAgenticSystem
                })
            });
            const data = await res.json();

            // If it was a new session (no selectedSession), the backend created one
            if (!selectedSession) {
                fetchSessions(selectedWorkspace); // Refresh sessions to get the new one
            }

            setMessages(prev => [...prev, {
                role: 'assistant',
                content: data.response,
                mode: data.mode_used,
                metadata: data.metadata
            }]);
        } catch (err) {
            setMessages(prev => [...prev, { role: 'assistant', content: "### ⚠️ Connection Error\n\nCould not reach the EKM backend." }]);
        } finally { setLoading(false); }
    };

    const runSleep = async () => {
        if (!selectedWorkspace) return;
        setLoading(true);
        try {
            await fetch(`${API_BASE}/sleep/${selectedWorkspace}`, { method: 'POST' });
            alert("Consolidation cycle completed.");
        } catch (err) { console.error("Sleep failure", err); }
        finally { setLoading(false); }
    };

    const runDeepResearch = async (query, maxIterations = 3, includeChainOfThoughtsOpt = includeChainOfThoughts, useAgenticSystemOpt = useAgenticSystem) => {
        if (!selectedWorkspace || !query.trim()) return;

        setResearchLoading(true);
        try {
            const res = await fetch(`${API_BASE}/deep_research/${selectedWorkspace}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    workspace_id: selectedWorkspace,
                    query: query,
                    max_iterations: maxIterations
                })
            });

            const data = await res.json();

            if (data.status === "success") {
                // Add the research result as a new message
                const researchMsg = {
                    role: 'assistant',
                    content: `Deep Research Completed: Generated a comprehensive research document based on your query "${query}"`,
                    mode: 'deep_research',
                    showChainOfThoughts: false, // Initialize the toggle state
                    metadata: {
                        is_research: true,
                        latex_available: !!data.latex_content,
                        latex_content: data.latex_content,  // Store the content for download
                        agentic_data: data.agentic_data,
                        chain_of_thoughts: data.agentic_data?.chain_of_thoughts
                    }
                };

                setMessages(prev => [...prev, researchMsg]);
            } else {
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: `### ⚠️ Deep Research Error\n\n${data.latex_content || "An error occurred during deep research."}`
                }]);
            }
        } catch (err) {
            console.error("Deep research failure", err);
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `### ⚠️ Deep Research Error\n\nFailed to perform deep research: ${err.message}`
            }]);
        } finally {
            setResearchLoading(false);
        }
    };

    const fetchWorkspaceTasks = async (workspaceId) => {
        if (!workspaceId) return;

        try {
            const res = await fetch(`${API_BASE}/tasks/workspace/${workspaceId}`);
            const tasks = await res.json();
            setActiveTasks(tasks);
        } catch (err) {
            console.error("Failed to fetch tasks", err);
        }
    };

    const currentWorkspace = workspaces.find(w => w.id === selectedWorkspace);
    const currentSession = sessions.find(s => s.id === selectedSession);

    return (
        <div style={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden', color: '#e2e8f0' }}>
            <div className="mesh-gradient" />

            {/* ── Sidebar ── */}
            <motion.aside
                initial={false}
                animate={{ width: isSidebarOpen ? 280 : 0, opacity: isSidebarOpen ? 1 : 0 }}
                transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                className="glass-panel"
                style={{
                    borderRight: '1px solid rgba(255,255,255,0.06)',
                    display: 'flex', flexDirection: 'column', zIndex: 20, overflow: 'hidden',
                    flexShrink: 0, minWidth: 0,
                }}
            >
                {/* Logo & Workspace Dropdown */}
                <div style={{ padding: '16px 16px 12px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                        <div style={{
                            width: 32, height: 32, borderRadius: 8, background: '#2563eb',
                            display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                        }}>
                            <Cpu size={16} color="#fff" />
                        </div>
                        <span style={{ fontWeight: 700, fontSize: 15, letterSpacing: '-0.02em', whiteSpace: 'nowrap' }}>EKM Desktop</span>
                    </div>

                    {/* Workspace Selector */}
                    <div style={{ position: 'relative' }}>
                        <select
                            value={selectedWorkspace}
                            onChange={(e) => setSelectedWorkspace(e.target.value)}
                            style={{
                                width: '100%', padding: '9px 12px 9px 36px', borderRadius: 10,
                                background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
                                color: '#f1f5f9', fontSize: 13, fontWeight: 500, outline: 'none',
                                appearance: 'none', cursor: 'pointer',
                            }}
                        >
                            {workspaces.map(w => (
                                <option key={w.id} value={w.id} style={{ background: '#0f172a' }}>{w.name}</option>
                            ))}
                        </select>
                        <Layers size={14} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: '#64748b' }} />
                        <button
                            onClick={() => setShowWorkspaceModal(true)}
                            style={{
                                position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)',
                                background: 'rgba(255,255,255,0.08)', border: 'none', borderRadius: 6,
                                width: 22, height: 22, display: 'flex', alignItems: 'center', justifyContent: 'center',
                                color: '#fff', cursor: 'pointer'
                            }}
                        >
                            <Plus size={12} />
                        </button>
                    </div>
                </div>

                {/* New Chat Button */}
                <div style={{ padding: '0 16px 16px' }}>
                    <button
                        onClick={createSession}
                        className="glass-btn group"
                        style={{
                            width: '100%', display: 'flex', alignItems: 'center', gap: 8,
                            padding: '10px 14px', borderRadius: 10,
                            color: '#fff', fontSize: 13, fontWeight: 600, cursor: 'pointer',
                        }}
                    >
                        <Plus size={16} className="group-hover:rotate-90 transition-transform duration-300" /> New Chat
                    </button>
                </div>

                {/* Sessions List */}
                <nav style={{ flex: 1, overflowY: 'auto', padding: '0 10px' }}>
                    <div style={{ padding: '4px 12px 8px', fontSize: 10, fontWeight: 700, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                        Recent Chats
                    </div>
                    {sessions.map(s => {
                        const active = selectedSession === s.id;
                        return (
                            <div key={s.id} style={{ position: 'relative', group: 'true' }}>
                                <button
                                    onClick={() => setSelectedSession(s.id)}
                                    style={{
                                        width: '100%', display: 'flex', alignItems: 'center', gap: 10,
                                        padding: '10px 12px', borderRadius: 10, marginBottom: 2,
                                        background: active ? 'rgba(59,130,246,0.12)' : 'transparent',
                                        border: 'none',
                                        color: active ? '#60a5fa' : '#94a3b8',
                                        cursor: 'pointer', textAlign: 'left', fontSize: 13, fontWeight: 500,
                                        transition: 'all 0.15s',
                                        paddingRight: 36,
                                    }}
                                >
                                    <MessageSquare size={14} style={{ flexShrink: 0, opacity: active ? 1 : 0.6 }} />
                                    <span style={{
                                        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                                        fontSize: 12.5
                                    }} title={s.name}>
                                        {s.name}
                                    </span>
                                </button>
                                <button
                                    onClick={(e) => deleteSession(e, s.id)}
                                    className="delete-btn"
                                    style={{
                                        position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)',
                                        background: 'transparent', border: 'none', color: '#475569',
                                        padding: 4, borderRadius: 6, cursor: 'pointer',
                                        display: active ? 'flex' : 'none', // Show only on active or hover?
                                    }}
                                >
                                    <Trash2 size={12} />
                                </button>
                            </div>
                        );
                    })}
                    {sessions.length === 0 && (
                        <div style={{ padding: '20px 12px', textAlign: 'center', color: '#475569', fontSize: 11 }}>
                            No history found
                        </div>
                    )}
                </nav>

                {/* Tasks Panel */}
                <div style={{ padding: '12px 16px', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <Box size={14} color="#94a3b8" />
                            <span style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                                Tasks
                            </span>
                        </div>
                        <button
                            onClick={() => fetchWorkspaceTasks(selectedWorkspace)}
                            style={{
                                background: 'transparent', border: 'none', color: '#64748b',
                                padding: '2px 6px', borderRadius: 6, cursor: 'pointer',
                                fontSize: 10
                            }}
                        >
                            Refresh
                        </button>
                    </div>
                    <div style={{ maxHeight: 150, overflowY: 'auto' }}>
                        {activeTasks.length === 0 ? (
                            <div style={{ padding: '8px 0', textAlign: 'center', color: '#475569', fontSize: 11 }}>
                                No active tasks
                            </div>
                        ) : (
                            activeTasks.map(task => (
                                <div key={task.id} className="glass-card" style={{ marginBottom: 6, padding: '8px 10px', borderRadius: 8 }}>
                                    <div style={{
                                        display: 'flex', justifyContent: 'space-between',
                                        fontSize: 11, color: '#e2e8f0', fontWeight: 500
                                    }}>
                                        <span>{task.name}</span>
                                        <span style={{
                                            color: task.status === 'completed' ? '#22c55e' :
                                                task.status === 'failed' ? '#ef4444' :
                                                    '#f59e0b'
                                        }}>
                                            {task.status.toUpperCase()}
                                        </span>
                                    </div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
                                        <div style={{
                                            flex: 1, height: 4, background: 'rgba(255,255,255,0.1)',
                                            borderRadius: 2, overflow: 'hidden'
                                        }}>
                                            <div style={{
                                                width: `${task.progress * 100}%`, height: '100%',
                                                background: task.status === 'completed' ? '#22c55e' :
                                                    task.status === 'failed' ? '#ef4444' :
                                                        '#f59e0b',
                                                transition: 'width 0.3s ease'
                                            }} />
                                        </div>
                                        <span style={{ fontSize: 9, color: '#94a3b8' }}>
                                            {Math.round(task.progress * 100)}%
                                        </span>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>

                {/* Sidebar Footer */}
                <div style={{ padding: '12px 16px', borderTop: '1px solid rgba(255,255,255,0.05)', background: 'rgba(0,0,0,0.15)' }}>
                    <button
                        onClick={runSleep}
                        disabled={loading}
                        style={{
                            width: '100%', display: 'flex', alignItems: 'center', gap: 8,
                            padding: '8px 12px', background: 'rgba(255,255,255,0.03)',
                            border: '1px solid rgba(255,255,255,0.06)', borderRadius: 10,
                            color: '#94a3b8', fontSize: 12, fontWeight: 500, cursor: 'pointer',
                        }}
                    >
                        <Moon size={13} /> Consolidate (Sleep)
                    </button>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 12px 0', fontSize: 11, color: '#475569' }}>
                        <Activity size={12} color="#22c55e" /> Agent Running
                    </div>
                </div>
            </motion.aside>

            {/* ── Main Content ── */}
            <main style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'transparent', minWidth: 0, position: 'relative' }}>

                {/* Header */}
                <header
                    className="glass-panel"
                    style={{
                        height: 60, display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                        padding: '0 24px', borderBottom: '1px solid rgba(255,255,255,0.05)', flexShrink: 0, zIndex: 5,
                    }}
                >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <button
                            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                            style={{
                                width: 32, height: 32, borderRadius: 8, background: 'rgba(255,255,255,0.04)',
                                border: '1px solid rgba(255,255,255,0.08)', color: '#94a3b8', cursor: 'pointer',
                                display: 'flex', alignItems: 'center', justifyContent: 'center'
                            }}
                        >
                            <ChevronRight size={18} style={{ transition: 'transform 0.25s', transform: isSidebarOpen ? 'rotate(180deg)' : 'none' }} />
                        </button>
                        <div style={{ height: 16, width: 1, background: 'rgba(255,255,255,0.08)', margin: '0 4px' }} />
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <span style={{ fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>{persona.name}</span>
                            <span style={{ fontSize: 10, color: '#64748b' }}>{currentWorkspace?.name || 'No Workspace'}</span>
                        </div>
                    </div>

                    {consciousness && (
                        <motion.div
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            style={{
                                display: 'flex', alignItems: 'center', gap: 8, padding: '4px 12px', borderRadius: 20,
                                background: 'rgba(168,85,247,0.1)', border: '1px solid rgba(168,85,247,0.2)',
                                cursor: 'help'
                            }}
                            title={`Mood: ${consciousness.mood}\nFocus: ${consciousness.thought_summary}`}
                        >
                            <Sparkles size={12} color="#c084fc" />
                            <span style={{ fontSize: 10, fontWeight: 700, color: '#d8b4fe' }}>{consciousness?.mood?.toUpperCase() || 'STABLE'}</span>
                        </motion.div>
                    )}

                    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                        {/* View toggle */}
                        <div style={{
                            display: 'flex', borderRadius: 10, overflow: 'hidden',
                            border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0,0,0,0.2)',
                            padding: 2,
                        }}>
                            {[{ key: 'chat', icon: MessageSquare, label: 'Chat' }, { key: 'graph', icon: Box, label: 'Graph' }].map(v => (
                                <button
                                    key={v.key}
                                    onClick={() => setViewMode(v.key)}
                                    style={{
                                        display: 'flex', alignItems: 'center', gap: 6,
                                        padding: '6px 14px', border: 'none', cursor: 'pointer',
                                        fontSize: 11, fontWeight: 600, borderRadius: 8,
                                        background: viewMode === v.key ? 'rgba(59,130,246,0.15)' : 'transparent',
                                        color: viewMode === v.key ? '#60a5fa' : '#64748b',
                                        transition: 'all 0.15s',
                                    }}
                                >
                                    <v.icon size={13} />
                                    {v.label}
                                </button>
                            ))}
                        </div>

                        {/* Research Tools */}
                        <div style={{
                            display: 'flex', borderRadius: 10, overflow: 'hidden',
                            border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0,0,0,0.2)',
                            padding: 2,
                        }}>
                            <button
                                onClick={() => {
                                    if (input.trim()) {
                                        setResearchQuery(input);
                                        setShowResearchModal(true);
                                    } else {
                                        alert("Please enter a query for deep research");
                                    }
                                }}
                                disabled={researchLoading}
                                style={{
                                    display: 'flex', alignItems: 'center', gap: 6,
                                    padding: '6px 12px', border: 'none', cursor: researchLoading ? 'wait' : 'pointer',
                                    fontSize: 11, fontWeight: 600, borderRadius: 8,
                                    background: researchLoading ? 'rgba(245, 158, 11, 0.3)' : 'transparent',
                                    color: researchLoading ? '#fbbf24' : '#fbbf24',
                                    transition: 'all 0.15s',
                                }}
                                title="Run deep research on your query"
                            >
                                {researchLoading ? (
                                    <div style={{ display: 'flex', gap: 5 }}>
                                        {[0, 0.15, 0.3].map(delay => (
                                            <div
                                                key={delay}
                                                style={{
                                                    width: 4, height: 4, borderRadius: '50%',
                                                    background: '#fbbf24',
                                                    animation: `pulse 1.5s infinite ${delay}s`
                                                }}
                                            />
                                        ))}
                                    </div>
                                ) : (
                                    <>
                                        <Database size={13} />
                                        Deep Research
                                    </>
                                )}
                            </button>
                        </div>

                        <div style={{
                            display: 'flex', alignItems: 'center', gap: 6, padding: '6px 12px', borderRadius: 20,
                            background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.15)',
                        }}>
                            <Sparkles size={12} color="#60a5fa" />
                            <span style={{ fontSize: 11, fontWeight: 600, color: '#93c5fd' }}>Gemini 1.5 Pro</span>
                        </div>
                        <button
                            onClick={() => setShowSettingsModal(true)}
                            style={{
                                width: 34, height: 34, borderRadius: 10, background: 'rgba(255,255,255,0.03)',
                                border: '1px solid rgba(255,255,255,0.06)', color: '#94a3b8', cursor: 'pointer',
                                display: 'flex', alignItems: 'center', justifyContent: 'center'
                            }}
                        >
                            <Settings size={17} />
                        </button>
                    </div>
                </header>

                {/* ── View: Chat or Graph ── */}
                {viewMode === 'graph' ? (
                    <MeshVisualization workspaceId={selectedWorkspace} />
                ) : (
                    <>
                        {/* Chat Messages */}
                        <div style={{ flex: 1, overflowY: 'auto', padding: '24px 24px 16px' }}>
                            <div style={{ maxWidth: 800, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 20 }}>

                                {/* Empty State */}
                                {messages.length === 0 && (
                                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', padding: '80px 16px 40px' }}>
                                        <motion.div
                                            initial={{ scale: 0.8, opacity: 0 }}
                                            animate={{ scale: 1, opacity: 1 }}
                                            style={{
                                                width: 64, height: 64, borderRadius: 20,
                                                background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.15)',
                                                display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 24,
                                                boxShadow: '0 0 20px rgba(59,130,246,0.1)'
                                            }}
                                        >
                                            <Cpu size={32} color="#3b82f6" />
                                        </motion.div>
                                        <h2 style={{ fontSize: 24, fontWeight: 800, color: '#f8fafc', marginBottom: 8, letterSpacing: '-0.02em' }}>
                                            The Episodic Knowledge Mesh
                                        </h2>
                                        <p style={{ fontSize: 14, color: '#94a3b8', maxWidth: 420, lineHeight: 1.6 }}>
                                            Select a workspace above and start a conversation. I'll automatically retrieve context from your Episodic Memory and Conceptual Knowledge.
                                        </p>
                                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 40, width: '100%', maxWidth: 520 }}>
                                            {[
                                                { t: 'Project Overview', d: 'What is this codebase about?' },
                                                { t: 'Recent Changes', d: 'What did we implement last?' },
                                                { t: 'Query Knowledge', d: 'Explain the EKM architecture.' },
                                                { t: 'Analyze Logic', d: 'How does Focus Buffer work?' }
                                            ].map(item => (
                                                <button
                                                    key={item.t}
                                                    onClick={() => {
                                                        setInput(item.d);
                                                        textareaRef.current?.focus();
                                                    }}
                                                    style={{
                                                        padding: '16px', borderRadius: 16, textAlign: 'left', cursor: 'pointer',
                                                        background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)',
                                                        color: '#f1f5f9', fontSize: 13, transition: 'all 0.2s',
                                                    }}
                                                    onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(59,130,246,0.3)'; e.currentTarget.style.background = 'rgba(59,130,246,0.05)'; }}
                                                    onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.05)'; e.currentTarget.style.background = 'rgba(255,255,255,0.02)'; }}
                                                >
                                                    <div style={{ fontWeight: 700, marginBottom: 4, color: '#60a5fa', fontSize: 11, textTransform: 'uppercase' }}>{item.t}</div>
                                                    <div style={{ color: '#94a3b8', fontSize: 13 }}>{item.d}</div>
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Messages */}
                                <AnimatePresence>
                                    {messages.map((msg, idx) => (
                                        <motion.div
                                            key={idx}
                                            initial={{ opacity: 0, y: 12 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ duration: 0.3 }}
                                            style={{ display: 'flex', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start' }}
                                        >
                                            <div
                                                className={msg.role === 'assistant' ? 'glass-card' : ''}
                                                style={{
                                                    maxWidth: '85%', borderRadius: 20, padding: '16px 20px',
                                                    ...(msg.role === 'user'
                                                        ? { background: '#3b82f6', color: '#fff', boxShadow: '0 8px 24px rgba(59, 130, 246, 0.25)' }
                                                        : {}
                                                    ),
                                                }}
                                            >
                                                {msg.role === 'assistant' && (
                                                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                                                        <div style={{ width: 20, height: 20, borderRadius: 6, background: '#2563eb', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                                            <Cpu size={12} color="#fff" />
                                                        </div>
                                                        <span style={{
                                                            padding: '2px 8px', borderRadius: 20, fontSize: 10, fontWeight: 700,
                                                            background: 'rgba(59,130,246,0.12)', color: '#60a5fa', border: '1px solid rgba(59,130,246,0.2)'
                                                        }}>
                                                            {msg.mode_used?.toUpperCase() || 'HYBRID'}
                                                        </span>
                                                        {msg.metadata?.focus_buffer_size > 0 && (
                                                            <span style={{
                                                                display: 'flex', alignItems: 'center', gap: 4,
                                                                padding: '2px 8px', borderRadius: 20, fontSize: 10, fontWeight: 700,
                                                                background: 'rgba(168,85,247,0.12)', color: '#c084fc', border: '1px solid rgba(168,85,247,0.2)'
                                                            }}>
                                                                <RefreshCw size={10} /> GROUNDED
                                                            </span>
                                                        )}
                                                    </div>
                                                )}
                                                <div className="prose prose-invert" style={{ fontSize: 14, lineHeight: 1.7, maxWidth: 'none', color: msg.role === 'user' ? '#fff' : '#cbd5e1' }}>
                                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                                                </div>
                                                {msg.chain_of_thoughts && (
                                                    <div style={{
                                                        marginTop: '16px',
                                                        padding: '12px',
                                                        borderRadius: '8px',
                                                        background: 'rgba(168, 85, 247, 0.08)',
                                                        border: '1px solid rgba(168, 85, 247, 0.15)',
                                                    }}>
                                                        <div style={{
                                                            display: 'flex',
                                                            alignItems: 'center',
                                                            gap: '8px',
                                                            marginBottom: '8px',
                                                            fontSize: '12px',
                                                            fontWeight: '700',
                                                            color: '#c084fc',
                                                        }}>
                                                            <Sparkles size={12} />
                                                            <span>CHAIN OF THOUGHTS</span>
                                                        </div>
                                                        <div style={{
                                                            fontSize: '13px',
                                                            lineHeight: '1.6',
                                                            color: '#d1d5db',
                                                            whiteSpace: 'pre-wrap',
                                                        }}>
                                                            <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.chain_of_thoughts}</ReactMarkdown>
                                                        </div>
                                                    </div>
                                                )}
                                                {msg.metadata?.use_agentic_system && msg.metadata?.evaluations && (
                                                    <div style={{
                                                        marginTop: '16px',
                                                        padding: '12px',
                                                        borderRadius: '8px',
                                                        background: 'rgba(59, 130, 246, 0.08)',
                                                        border: '1px solid rgba(59, 130, 246, 0.15)',
                                                    }}>
                                                        <div style={{
                                                            display: 'flex',
                                                            alignItems: 'center',
                                                            gap: '8px',
                                                            marginBottom: '8px',
                                                            fontSize: '12px',
                                                            fontWeight: '700',
                                                            color: '#3b82f6',
                                                        }}>
                                                            <Cpu size={12} />
                                                            <span>AGENTIC SYSTEM EVALUATION</span>
                                                        </div>
                                                        <div style={{
                                                            fontSize: '12px',
                                                            lineHeight: '1.5',
                                                            color: '#d1d5db',
                                                        }}>
                                                            <div>Iterations completed: {msg.metadata.iterations_completed}</div>
                                                            {msg.metadata.evaluations.map((evalItem, idx) => (
                                                                <div key={idx} style={{ marginTop: '8px' }}>
                                                                    <div style={{ fontWeight: '600', color: '#93c5fd' }}>Iteration {evalItem.iteration}:</div>
                                                                    <div>Query: {evalItem.query}</div>
                                                                    <div>Ready: {evalItem.evaluation.ready ? 'Yes' : 'No'}</div>
                                                                    {evalItem.evaluation.follow_up_query && (
                                                                        <div>Follow-up: {evalItem.evaluation.follow_up_query}</div>
                                                                    )}
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </div>
                                                )}
                                                {msg.metadata?.is_research && (
                                                    <div style={{
                                                        marginTop: '16px',
                                                        padding: '16px',
                                                        borderRadius: '12px',
                                                        background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%)',
                                                        border: '1px solid rgba(245, 158, 11, 0.2)',
                                                        boxShadow: '0 4px 20px rgba(245, 158, 11, 0.08)',
                                                    }}>
                                                        <div style={{
                                                            display: 'flex',
                                                            alignItems: 'center',
                                                            justifyContent: 'space-between',
                                                            marginBottom: '12px',
                                                        }}>
                                                            <div style={{
                                                                display: 'flex',
                                                                alignItems: 'center',
                                                                gap: '8px',
                                                            }}>
                                                                <div style={{
                                                                    width: 28,
                                                                    height: 28,
                                                                    borderRadius: '8px',
                                                                    background: 'rgba(245, 158, 11, 0.15)',
                                                                    display: 'flex',
                                                                    alignItems: 'center',
                                                                    justifyContent: 'center',
                                                                }}>
                                                                    <Database size={14} color="#fbbf24" />
                                                                </div>
                                                                <div>
                                                                    <div style={{
                                                                        fontSize: '13px',
                                                                        fontWeight: '700',
                                                                        color: '#fbbf24',
                                                                        lineHeight: '1.4',
                                                                    }}>
                                                                        DEEP RESEARCH COMPLETED
                                                                    </div>
                                                                    <div style={{
                                                                        fontSize: '10px',
                                                                        color: '#fbbf24',
                                                                        opacity: 0.7,
                                                                        fontWeight: '500',
                                                                    }}>
                                                                        Comprehensive analysis generated
                                                                    </div>
                                                                </div>
                                                            </div>
                                                            <div style={{
                                                                display: 'flex',
                                                                gap: '6px',
                                                            }}>
                                                                {msg.metadata?.agentic_data?.metadata?.iterations_completed && (
                                                                    <div style={{
                                                                        padding: '4px 8px',
                                                                        borderRadius: '6px',
                                                                        background: 'rgba(59, 130, 246, 0.15)',
                                                                        border: '1px solid rgba(59, 130, 246, 0.25)',
                                                                        fontSize: '9px',
                                                                        fontWeight: '700',
                                                                        color: '#93c5fd',
                                                                    }}>
                                                                        {msg.metadata.agentic_data.metadata.iterations_completed} ITERATIONS
                                                                    </div>
                                                                )}
                                                                {msg.metadata?.chain_of_thoughts && (
                                                                    <div style={{
                                                                        padding: '4px 8px',
                                                                        borderRadius: '6px',
                                                                        background: 'rgba(168, 85, 247, 0.15)',
                                                                        border: '1px solid rgba(168, 85, 247, 0.25)',
                                                                        fontSize: '9px',
                                                                        fontWeight: '700',
                                                                        color: '#d8b4fe',
                                                                    }}>
                                                                        REASONING
                                                                    </div>
                                                                )}
                                                            </div>
                                                        </div>
                                                        <div style={{
                                                            fontSize: '13px',
                                                            lineHeight: '1.6',
                                                            color: '#fde68a',
                                                            marginBottom: '12px',
                                                        }}>
                                                            <p>This response was generated through comprehensive deep research using agentic systems and chain of thoughts reasoning.</p>
                                                        </div>
                                                        <div style={{
                                                            display: 'flex',
                                                            gap: '8px',
                                                            flexWrap: 'wrap',
                                                        }}>
                                                            {msg.metadata.latex_available && (
                                                                <button
                                                                    onClick={() => {
                                                                        // Trigger LaTeX document download
                                                                        if (msg.metadata.latex_content) {
                                                                            const blob = new Blob([msg.metadata.latex_content], { type: 'text/plain' });
                                                                            const url = window.URL.createObjectURL(blob);
                                                                            const a = document.createElement('a');
                                                                            a.href = url;
                                                                            a.download = `research_paper_${Date.now()}.tex`;
                                                                            document.body.appendChild(a);
                                                                            a.click();
                                                                            window.URL.revokeObjectURL(url);
                                                                            document.body.removeChild(a);
                                                                        }
                                                                    }}
                                                                    style={{
                                                                        padding: '8px 16px',
                                                                        borderRadius: '8px',
                                                                        background: 'rgba(245, 158, 11, 0.2)',
                                                                        border: '1px solid rgba(245, 158, 11, 0.3)',
                                                                        color: '#fbbf24',
                                                                        cursor: 'pointer',
                                                                        fontSize: '12px',
                                                                        fontWeight: '600',
                                                                        display: 'flex',
                                                                        alignItems: 'center',
                                                                        gap: '6px',
                                                                        transition: 'all 0.2s',
                                                                    }}
                                                                    onMouseOver={(e) => {
                                                                        e.target.style.background = 'rgba(245, 158, 11, 0.3)';
                                                                        e.target.style.transform = 'translateY(-1px)';
                                                                    }}
                                                                    onMouseOut={(e) => {
                                                                        e.target.style.background = 'rgba(245, 158, 11, 0.2)';
                                                                        e.target.style.transform = 'translateY(0)';
                                                                    }}
                                                                >
                                                                    <Database size={14} />
                                                                    Download Report
                                                                </button>
                                                            )}
                                                            {msg.metadata?.chain_of_thoughts && (
                                                                <button
                                                                    onClick={() => {
                                                                        // Toggle visibility of chain of thoughts
                                                                        setMessages(prev => prev.map(m =>
                                                                            m === msg ? { ...m, showChainOfThoughts: !m.showChainOfThoughts } : m
                                                                        ));
                                                                    }}
                                                                    style={{
                                                                        padding: '8px 16px',
                                                                        borderRadius: '8px',
                                                                        background: 'rgba(168, 85, 247, 0.15)',
                                                                        border: '1px solid rgba(168, 85, 247, 0.25)',
                                                                        color: '#c084fc',
                                                                        cursor: 'pointer',
                                                                        fontSize: '12px',
                                                                        fontWeight: '600',
                                                                        display: 'flex',
                                                                        alignItems: 'center',
                                                                        gap: '6px',
                                                                        transition: 'all 0.2s',
                                                                    }}
                                                                    onMouseOver={(e) => {
                                                                        e.target.style.background = 'rgba(168, 85, 247, 0.25)';
                                                                        e.target.style.transform = 'translateY(-1px)';
                                                                    }}
                                                                    onMouseOut={(e) => {
                                                                        e.target.style.background = 'rgba(168, 85, 247, 0.15)';
                                                                        e.target.style.transform = 'translateY(0)';
                                                                    }}
                                                                >
                                                                    <Sparkles size={14} />
                                                                    Show Reasoning
                                                                </button>
                                                            )}
                                                        </div>
                                                        {msg.showChainOfThoughts && msg.metadata?.chain_of_thoughts && (
                                                            <div style={{
                                                                marginTop: '12px',
                                                                padding: '12px',
                                                                borderRadius: '8px',
                                                                background: 'rgba(168, 85, 247, 0.08)',
                                                                border: '1px solid rgba(168, 85, 247, 0.15)',
                                                            }}>
                                                                <div style={{
                                                                    display: 'flex',
                                                                    alignItems: 'center',
                                                                    gap: '8px',
                                                                    marginBottom: '8px',
                                                                    fontSize: '12px',
                                                                    fontWeight: '700',
                                                                    color: '#c084fc',
                                                                }}>
                                                                    <Sparkles size={12} />
                                                                    <span>CHAIN OF THOUGHTS</span>
                                                                </div>
                                                                <div style={{
                                                                    fontSize: '13px',
                                                                    lineHeight: '1.6',
                                                                    color: '#d1d5db',
                                                                    whiteSpace: 'pre-wrap',
                                                                }}>
                                                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.metadata.chain_of_thoughts}</ReactMarkdown>
                                                                </div>
                                                            </div>
                                                        )}
                                                    </div>
                                                )}
                                            </div>
                                        </motion.div>
                                    ))}
                                </AnimatePresence>

                                {/* Thinking Indicator */}
                                {loading && (
                                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ display: 'flex', justifyContent: 'flex-start' }}>
                                        <div className="glass" style={{ borderRadius: 16, padding: '12px 20px', display: 'flex', gap: 12, alignItems: 'center', border: '1px solid rgba(255,255,255,0.08)' }}>
                                            <div style={{ display: 'flex', gap: 5 }}>
                                                {[0, 0.15, 0.3].map(delay => (
                                                    <motion.div
                                                        key={delay}
                                                        animate={{ y: [0, -3, 0], opacity: [0.4, 1, 0.4] }}
                                                        transition={{ duration: 0.6, repeat: Infinity, delay }}
                                                        style={{ width: 6, height: 6, borderRadius: '50%', background: '#60a5fa' }}
                                                    />
                                                ))}
                                            </div>
                                            <span style={{ fontSize: 12, fontWeight: 600, color: '#64748b', letterSpacing: '0.02em' }}>
                                                Processing Mesh…
                                            </span>
                                        </div>
                                    </motion.div>
                                )}
                                <div ref={chatEndRef} />
                            </div>
                        </div>

                        {/* ── Input Area ── */}
                        <div style={{ padding: '16px 24px 24px', flexShrink: 0 }}>
                            <div style={{ maxWidth: 800, margin: '0 auto' }}>
                                <div className="glass-panel" style={{
                                    padding: '8px', borderRadius: 20, border: '1px solid rgba(255,255,255,0.08)',
                                    background: 'rgba(10,10,14,0.6)', backdropFilter: 'blur(20px)',
                                    display: 'flex', gap: 8, alignItems: 'flex-end',
                                    boxShadow: '0 10px 40px rgba(0,0,0,0.4)'
                                }}>
                                    <button
                                        onClick={() => fileInputRef.current.click()}
                                        style={{
                                            width: 44, height: 44, borderRadius: 14,
                                            background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)',
                                            color: '#94a3b8', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
                                            transition: 'all 0.2s', flexShrink: 0,
                                        }}
                                        onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.06)'}
                                        onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,255,255,0.03)'}
                                    >
                                        <Upload size={20} />
                                    </button>
                                    <div style={{ flex: 1, display: 'flex', alignItems: 'center' }}>
                                        <textarea
                                            ref={textareaRef}
                                            autoFocus
                                            value={input}
                                            onChange={(e) => setInput(e.target.value)}
                                            onKeyDown={(e) => {
                                                if (e.key === 'Enter' && !e.shiftKey) {
                                                    e.preventDefault();
                                                    sendMessage();
                                                }
                                            }}
                                            placeholder="Message the agent..."
                                            className="glass-input"
                                            style={{
                                                width: '100%',
                                                background: 'transparent',
                                                border: 'none',
                                                padding: '12px 12px',
                                                fontSize: '14px',
                                                color: '#f1f5f9',
                                                outline: 'none',
                                                resize: 'none',
                                                minHeight: '44px',
                                                maxHeight: '200px',
                                                fontFamily: 'inherit',
                                                lineHeight: '1.5',
                                            }}
                                            onInput={(e) => {
                                                e.target.style.height = 'auto';
                                                e.target.style.height = e.target.scrollHeight + 'px';
                                            }}
                                        />
                                    </div>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginRight: 8 }}>
                                        <button
                                            onClick={() => setIncludeChainOfThoughts(!includeChainOfThoughts)}
                                            style={{
                                                width: 32, height: 32, borderRadius: 10,
                                                background: includeChainOfThoughts ? 'rgba(168, 85, 247, 0.2)' : 'rgba(255,255,255,0.03)',
                                                border: '1px solid ' + (includeChainOfThoughts ? 'rgba(168, 85, 247, 0.4)' : 'rgba(255,255,255,0.06)'),
                                                color: includeChainOfThoughts ? '#c084fc' : '#94a3b8',
                                                cursor: 'pointer',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                transition: 'all 0.2s',
                                                flexShrink: 0,
                                            }}
                                            title={includeChainOfThoughts ? "Chain of thoughts enabled" : "Enable chain of thoughts"}
                                        >
                                            <Sparkles size={16} />
                                        </button>
                                        <span style={{
                                            fontSize: '8px',
                                            textAlign: 'center',
                                            color: includeChainOfThoughts ? '#c084fc' : '#64748b',
                                            fontWeight: 'bold',
                                            textTransform: 'uppercase',
                                            letterSpacing: '0.5px'
                                        }}>
                                            CoT
                                        </span>
                                    </div>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginRight: 8 }}>
                                        <button
                                            onClick={() => setUseAgenticSystem(!useAgenticSystem)}
                                            style={{
                                                width: 32, height: 32, borderRadius: 10,
                                                background: useAgenticSystem ? 'rgba(59, 130, 246, 0.2)' : 'rgba(255,255,255,0.03)',
                                                border: '1px solid ' + (useAgenticSystem ? 'rgba(59, 130, 246, 0.4)' : 'rgba(255,255,255,0.06)'),
                                                color: useAgenticSystem ? '#3b82f6' : '#94a3b8',
                                                cursor: 'pointer',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                transition: 'all 0.2s',
                                                flexShrink: 0,
                                            }}
                                            title={useAgenticSystem ? "Agentic system enabled" : "Enable agentic system"}
                                        >
                                            <Cpu size={16} />
                                        </button>
                                        <span style={{
                                            fontSize: '8px',
                                            textAlign: 'center',
                                            color: useAgenticSystem ? '#3b82f6' : '#64748b',
                                            fontWeight: 'bold',
                                            textTransform: 'uppercase',
                                            letterSpacing: '0.5px'
                                        }}>
                                            Agentic
                                        </span>
                                    </div>
                                    <button
                                        onClick={sendMessage}
                                        disabled={loading || !input.trim()}
                                        style={{
                                            width: 44, height: 44, borderRadius: 14,
                                            border: 'none', cursor: input.trim() && !loading ? 'pointer' : 'default',
                                            background: input.trim() && !loading ? '#2563eb' : 'rgba(100,116,139,0.1)',
                                            color: input.trim() && !loading ? '#fff' : '#475569',
                                            transition: 'all 0.2s', display: 'flex', alignItems: 'center', justifyContent: 'center',
                                            flexShrink: 0,
                                        }}
                                    >
                                        <Send size={20} />
                                    </button>
                                </div>

                                <input type="file" multiple ref={fileInputRef} onChange={handleFileUpload} style={{ display: 'none' }} />

                                <div style={{
                                    display: 'flex', justifyContent: 'space-between',
                                    fontSize: 10, color: '#475569', padding: '8px 8px 0', fontWeight: 500,
                                }}>
                                    <div style={{ display: 'flex', gap: 16 }}>
                                        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                                            <kbd style={{ background: 'rgba(255,255,255,0.05)', padding: '1px 4px', borderRadius: 4, fontSize: 9 }}>Enter</kbd> to send
                                        </span>
                                        <span>Grounded in {selectedWorkspace.slice(0, 8)}</span>
                                    </div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                                        <RefreshCw size={10} className={loading ? 'animate-spin' : ''} />
                                        <span>EKM Sidecar connected</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </>
                )}

                {/* New Workspace Modal */}
                <AnimatePresence>
                    {showWorkspaceModal && (
                        <div style={{
                            position: 'fixed', inset: 0, zIndex: 100,
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(8px)',
                        }}>
                            <motion.div
                                initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }}
                                className="glass-panel"
                                style={{
                                    width: '100%', maxWidth: 420, padding: 32, borderRadius: 24,
                                    border: '1px solid rgba(255,255,255,0.1)',
                                    boxShadow: '0 20px 50px rgba(0,0,0,0.5)'
                                }}
                            >
                                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
                                    <div style={{ width: 40, height: 40, borderRadius: 12, background: 'rgba(59,130,246,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                        <Layers size={20} color="#3b82f6" />
                                    </div>
                                    <div>
                                        <h3 style={{ fontSize: 18, fontWeight: 800, color: '#f8fafc', marginBottom: 2 }}>Create Workspace</h3>
                                        <p style={{ fontSize: 12, color: '#64748b' }}>A new context for your knowledge mesh.</p>
                                    </div>
                                </div>
                                <input
                                    autoFocus
                                    type="text"
                                    value={newWorkspaceName}
                                    onChange={(e) => setNewWorkspaceName(e.target.value)}
                                    onKeyDown={(e) => { if (e.key === 'Enter') createWorkspace(); }}
                                    placeholder="Enter workspace name..."
                                    className="glass-input"
                                    style={{
                                        width: '100%', background: 'rgba(255,255,255,0.03)',
                                        border: '1px solid rgba(255,255,255,0.08)', borderRadius: 12,
                                        padding: '12px 16px', fontSize: 14, color: '#f8fafc',
                                        marginBottom: 24, outline: 'none', fontFamily: 'inherit',
                                    }}
                                />
                                <div style={{ display: 'flex', gap: 12 }}>
                                    <button
                                        onClick={() => setShowWorkspaceModal(false)}
                                        className="glass-btn"
                                        style={{
                                            flex: 1, padding: '12px', borderRadius: 12,
                                            background: 'rgba(255,255,255,0.05)', border: 'none', color: '#cbd5e1',
                                            fontSize: 14, fontWeight: 600, cursor: 'pointer',
                                        }}
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        onClick={createWorkspace}
                                        disabled={!newWorkspaceName.trim()}
                                        className="bg-accent-primary hover:bg-accent-secondary text-white font-semibold"
                                        style={{
                                            flex: 1, padding: '12px', borderRadius: 12,
                                            background: '#2563eb', border: 'none', color: '#fff',
                                            fontSize: 14, fontWeight: 600, cursor: 'pointer',
                                            boxShadow: '0 4px 12px rgba(37,99,235,0.2)',
                                        }}
                                    >
                                        Create
                                    </button>
                                </div>
                            </motion.div>
                        </div>
                    )}
                </AnimatePresence>

                {/* EKM Settings Modal */}
                <AnimatePresence>
                    {showSettingsModal && (
                        <div style={{
                            position: 'fixed', inset: 0, zIndex: 100,
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(8px)',
                        }}>
                            <motion.div
                                initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }}
                                className="glass-panel"
                                style={{
                                    width: '100%', maxWidth: 480, padding: 32, borderRadius: 24,
                                    border: '1px solid rgba(255,255,255,0.1)',
                                    boxShadow: '0 20px 50px rgba(0,0,0,0.5)',
                                    maxHeight: '90vh', overflowY: 'auto'
                                }}
                            >
                                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 24 }}>
                                    <div style={{ width: 40, height: 40, borderRadius: 12, background: 'rgba(37,99,235,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                        <Settings size={20} color="#3b82f6" />
                                    </div>
                                    <div>
                                        <h3 style={{ fontSize: 18, fontWeight: 800, color: '#f8fafc', marginBottom: 2 }}>EKM Parameters</h3>
                                        <p style={{ fontSize: 12, color: '#64748b' }}>Configure the Knowledge Mesh for this workspace.</p>
                                    </div>
                                </div>

                                <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                                    {/* Semantic Threshold */}
                                    <div>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                                            <label style={{ fontSize: 12, fontWeight: 700, color: '#94a3b8' }}>Semantic Threshold</label>
                                            <span style={{ fontSize: 12, fontWeight: 700, color: '#60a5fa' }}>{settings.EKM_SEMANTIC_THRESHOLD}</span>
                                        </div>
                                        <input
                                            type="range" min="0" max="1" step="0.01"
                                            value={settings.EKM_SEMANTIC_THRESHOLD || 0.70}
                                            onChange={(e) => setSettings({ ...settings, EKM_SEMANTIC_THRESHOLD: parseFloat(e.target.value) })}
                                            style={{ width: '100%', accentColor: '#3b82f6' }}
                                        />
                                        <p style={{ fontSize: 10, color: '#475569', marginTop: 4 }}>Lower values retrieve more, higher values are more precise.</p>
                                    </div>

                                    {/* RL Learning Rate */}
                                    <div>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                                            <label style={{ fontSize: 12, fontWeight: 700, color: '#94a3b8' }}>RL Learning Rate</label>
                                            <span style={{ fontSize: 12, fontWeight: 700, color: '#c084fc' }}>{settings.RL_LEARNING_RATE}</span>
                                        </div>
                                        <input
                                            type="range" min="0" max="1" step="0.01"
                                            value={settings.RL_LEARNING_RATE || 0.1}
                                            onChange={(e) => setSettings({ ...settings, RL_LEARNING_RATE: parseFloat(e.target.value) })}
                                            style={{ width: '100%', accentColor: '#a855f7' }}
                                        />
                                    </div>

                                    {/* Chunk Sizes */}
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                                        <div>
                                            <label style={{ fontSize: 11, fontWeight: 700, color: '#475569', textTransform: 'uppercase', marginBottom: 6, display: 'block' }}>Min Chunk Size</label>
                                            <input
                                                type="number"
                                                value={settings.EKM_MIN_CHUNK_SIZE || 512}
                                                onChange={(e) => setSettings({ ...settings, EKM_MIN_CHUNK_SIZE: parseInt(e.target.value) })}
                                                className="glass-input"
                                                style={{ width: '100%', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, padding: '8px 12px', color: '#fff', fontSize: 13 }}
                                            />
                                        </div>
                                        <div>
                                            <label style={{ fontSize: 11, fontWeight: 700, color: '#475569', textTransform: 'uppercase', marginBottom: 6, display: 'block' }}>Max Chunk Size</label>
                                            <input
                                                type="number"
                                                value={settings.EKM_MAX_CHUNK_SIZE || 2048}
                                                onChange={(e) => setSettings({ ...settings, EKM_MAX_CHUNK_SIZE: parseInt(e.target.value) })}
                                            />
                                        </div>
                                    </div>
                                </div>

                                <div style={{ display: 'flex', gap: 12, marginTop: 32 }}>
                                    <button
                                        onClick={() => setShowSettingsModal(false)}
                                        style={{
                                            flex: 1, padding: '12px', borderRadius: 12,
                                            background: 'rgba(255,255,255,0.05)', border: 'none', color: '#cbd5e1',
                                            fontSize: 14, fontWeight: 600, cursor: 'pointer',
                                        }}
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        onClick={updateSettings}
                                        style={{
                                            flex: 1, padding: '12px', borderRadius: 12,
                                            background: '#2563eb', border: 'none', color: '#fff',
                                            fontSize: 14, fontWeight: 600, cursor: 'pointer',
                                            boxShadow: '0 4px 12px rgba(37,99,235,0.2)',
                                        }}
                                    >
                                        Save Changes
                                    </button>
                                </div>
                            </motion.div>
                        </div>
                    )}
                </AnimatePresence>
            </main>
            {/* ── Global Loading State / Splash Screen ── */}
            <AnimatePresence>
                {!isBackendReady && (
                    <motion.div
                        initial={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        style={{
                            position: 'fixed', inset: 0, zIndex: 9999,
                            background: '#050508',
                            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                            gap: 24
                        }}
                    >
                        <motion.div
                            animate={{
                                scale: [1, 1.1, 1],
                                opacity: [0.5, 1, 0.5],
                                rotate: [0, 90, 180, 270, 360]
                            }}
                            transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                            style={{
                                width: 80, height: 80, borderRadius: 24,
                                background: 'rgba(59,130,246,0.1)',
                                border: '2px solid #2563eb',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                boxShadow: '0 0 40px rgba(37,99,235,0.3)'
                            }}
                        >
                            <Cpu size={40} color="#2563eb" />
                        </motion.div>
                        <div style={{ textAlign: 'center' }}>
                            <h2 style={{ fontSize: 24, fontWeight: 800, color: '#f8fafc', marginBottom: 8 }}>Initializing EKM</h2>
                            <p style={{ fontSize: 14, color: '#64748b', display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'center' }}>
                                <RefreshCw size={14} className="animate-spin" />
                                {connectionError || "Establishing connection to Sidecar..."}
                            </p>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Research Modal */}
            <AnimatePresence>
                {showResearchModal && (
                    <div style={{
                        position: 'fixed', inset: 0, zIndex: 100,
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(8px)',
                    }}>
                        <motion.div
                            initial={{ scale: 0.95, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.95, opacity: 0 }}
                            className="glass"
                            style={{
                                width: '100%', maxWidth: 520, padding: 32, borderRadius: 24,
                                border: '1px solid rgba(255,255,255,0.1)',
                                boxShadow: '0 20px 50px rgba(0,0,0,0.5)'
                            }}
                        >
                            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 24 }}>
                                <div style={{ width: 48, height: 48, borderRadius: 12, background: 'rgba(245, 158, 11, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    <Database size={24} color="#fbbf24" />
                                </div>
                                <div>
                                    <h3 style={{ fontSize: 20, fontWeight: 800, color: '#f8fafc', marginBottom: 4 }}>Deep Research</h3>
                                    <p style={{ fontSize: 14, color: '#94a3b8' }}>Generate a comprehensive research document based on your query</p>
                                </div>
                            </div>

                            <div style={{ marginBottom: 24 }}>
                                <label style={{ display: 'block', fontSize: 12, fontWeight: 700, color: '#94a3b8', marginBottom: 8 }}>Research Query</label>
                                <div style={{
                                    width: '100%', background: 'rgba(255,255,255,0.03)',
                                    border: '1px solid rgba(255,255,255,0.08)', borderRadius: 12,
                                    padding: '12px 16px', fontSize: 14, color: '#f8fafc',
                                    marginBottom: 16, outline: 'none', fontFamily: 'inherit',
                                    minHeight: '80px',
                                    maxHeight: '120px',
                                    overflowY: 'auto'
                                }}>
                                    {researchQuery}
                                </div>
                            </div>

                            <div style={{ display: 'flex', gap: 12, marginBottom: 24 }}>
                                <div style={{ flex: 1 }}>
                                    <label style={{ display: 'block', fontSize: 12, fontWeight: 700, color: '#94a3b8', marginBottom: 8 }}>Iterations</label>
                                    <select
                                        ref={iterationsRef}
                                        defaultValue="3"
                                        style={{
                                            width: '100%', padding: '10px 14px', borderRadius: 10,
                                            background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
                                            color: '#f1f5f9', fontSize: 13, fontWeight: 500, outline: 'none',
                                            appearance: 'none', cursor: 'pointer',
                                        }}
                                    >
                                        <option value="1">1 Iteration</option>
                                        <option value="2">2 Iterations</option>
                                        <option value="3">3 Iterations (Recommended)</option>
                                        <option value="5">5 Iterations</option>
                                    </select>
                                </div>
                                <div style={{ flex: 1 }}>
                                    <label style={{ display: 'block', fontSize: 12, fontWeight: 700, color: '#94a3b8', marginBottom: 8 }}>Format</label>
                                    <select
                                        ref={formatRef}
                                        defaultValue="latex"
                                        style={{
                                            width: '100%', padding: '10px 14px', borderRadius: 10,
                                            background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
                                            color: '#f1f5f9', fontSize: 13, fontWeight: 500, outline: 'none',
                                            appearance: 'none', cursor: 'pointer',
                                        }}
                                    >
                                        <option value="latex">LaTeX Document</option>
                                        <option value="markdown">Markdown Report</option>
                                        <option value="pdf">PDF Report</option>
                                    </select>
                                </div>
                            </div>

                            <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 24 }}>
                                <div style={{ display: 'flex', alignItems: 'center' }}>
                                    <input
                                        type="checkbox"
                                        ref={includeChainOfThoughtsRef}
                                        id="includeChainOfThoughts"
                                        defaultChecked={includeChainOfThoughts}
                                        style={{ width: 18, height: 18, marginRight: 8 }}
                                    />
                                    <label htmlFor="includeChainOfThoughts" style={{ fontSize: 13, color: '#e2e8f0' }}>Include Chain of Thoughts</label>
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center' }}>
                                    <input
                                        type="checkbox"
                                        ref={useAgenticSystemRef}
                                        id="useAgenticSystem"
                                        defaultChecked={useAgenticSystem}
                                        style={{ width: 18, height: 18, marginRight: 8 }}
                                    />
                                    <label htmlFor="useAgenticSystem" style={{ fontSize: 13, color: '#e2e8f0' }}>Use Agentic System</label>
                                </div>
                            </div>

                            <div style={{ display: 'flex', gap: 12 }}>
                                <button
                                    onClick={() => setShowResearchModal(false)}
                                    style={{
                                        flex: 1, padding: '12px', borderRadius: 12,
                                        background: 'rgba(255,255,255,0.05)', border: 'none', color: '#cbd5e1',
                                        fontSize: 14, fontWeight: 600, cursor: 'pointer',
                                    }}
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={() => {
                                        const iterations = parseInt(iterationsRef.current?.value || 3) || 3;
                                        const format = formatRef.current?.value || 'latex';
                                        const includeChainOfThoughtsOpt = includeChainOfThoughtsRef.current?.checked ?? includeChainOfThoughts;
                                        const useAgenticSystemOpt = useAgenticSystemRef.current?.checked ?? useAgenticSystem;

                                        runDeepResearch(researchQuery, iterations, includeChainOfThoughtsOpt, useAgenticSystemOpt);
                                        setShowResearchModal(false);
                                    }}
                                    disabled={researchLoading}
                                    style={{
                                        flex: 1, padding: '12px', borderRadius: 12,
                                        background: '#f59e0b', border: 'none', color: '#fff',
                                        fontSize: 14, fontWeight: 600, cursor: researchLoading ? 'wait' : 'pointer',
                                        boxShadow: '0 4px 12px rgba(245, 158, 11, 0.3)',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        gap: '8px'
                                    }}
                                >
                                    {researchLoading ? (
                                        <>
                                            <div style={{ display: 'flex', gap: 4 }}>
                                                {[0, 0.15, 0.3].map(delay => (
                                                    <div
                                                        key={delay}
                                                        style={{
                                                            width: 4, height: 4, borderRadius: '50%',
                                                            background: '#fff',
                                                            animation: 'pulse 1.5s infinite',
                                                            animationDelay: `${delay}s`
                                                        }}
                                                    />
                                                ))}
                                            </div>
                                            Generating...
                                        </>
                                    ) : (
                                        <>
                                            <Database size={16} />
                                            Generate Report
                                        </>
                                    )}
                                </button>
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </div>
    );
}

export default App;
