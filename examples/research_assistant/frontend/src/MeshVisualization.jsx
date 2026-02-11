import React, { useRef, useState, useEffect, useMemo, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Html, Stars, Line } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';

const API_BASE = 'http://127.0.0.1:8000';

/* ─────────── Color palette ─────────── */
const COLORS = {
    aku: new THREE.Color(0x3b82f6),
    akuArchived: new THREE.Color(0x6b7280),
    gku: new THREE.Color(0xfbbf24),
    semantic: new THREE.Color(0x60a5fa),
    causal: new THREE.Color(0xf97316),
    temporal: new THREE.Color(0x34d399),
    membership: new THREE.Color(0xa78bfa),
};
const COLOR_HEX = {
    aku: '#3b82f6', akuArchived: '#6b7280', gku: '#fbbf24',
    semantic: '#60a5fa', causal: '#f97316', temporal: '#34d399', membership: '#a78bfa',
};

/* ─────────── Animated AKU sphere ─────────── */
function AKUNode({ pos, label, archived, id, isSelected, onClick, onHover }) {
    const ref = useRef();
    const [hovered, setHovered] = useState(false);
    const color = archived ? COLORS.akuArchived : COLORS.aku;

    useFrame((_, delta) => {
        if (ref.current) {
            ref.current.scale.lerp(
                new THREE.Vector3(hovered ? 1.6 : 1, hovered ? 1.6 : 1, hovered ? 1.6 : 1),
                delta * 6
            );
        }
    });

    return (
        <group position={pos}>
            <mesh
                ref={ref}
                onClick={(e) => { e.stopPropagation(); onClick(id, pos); }}
                onPointerOver={(e) => { e.stopPropagation(); setHovered(true); onHover(id, label); }}
                onPointerOut={(e) => { e.stopPropagation(); setHovered(false); onHover(null, null); }}
            >
                <sphereGeometry args={[0.12, 24, 24]} />
                <meshStandardMaterial
                    color={color}
                    emissive={color}
                    emissiveIntensity={isSelected ? 1.2 : hovered ? 0.8 : 0.3}
                    roughness={0.3}
                    metalness={0.6}
                />
            </mesh>
            {(hovered || isSelected) && (
                <mesh>
                    <ringGeometry args={[0.18, 0.22, 32]} />
                    <meshBasicMaterial color={color} transparent opacity={0.4} side={THREE.DoubleSide} />
                </mesh>
            )}
        </group>
    );
}

/* ─────────── Animated GKU sphere ─────────── */
function GKUNode({ pos, label, id, isSelected, onClick, onHover, pattern }) {
    const ref = useRef();
    const glowRef = useRef();
    const [hovered, setHovered] = useState(false);

    useFrame(({ clock }, delta) => {
        const t = clock.getElapsedTime();
        const pulse = 1 + Math.sin(t * 2) * 0.06;
        if (ref.current) {
            const target = hovered ? 1.4 * pulse : pulse;
            ref.current.scale.lerp(new THREE.Vector3(target, target, target), delta * 5);
        }
        if (glowRef.current) {
            glowRef.current.rotation.y += delta * 0.3;
            glowRef.current.rotation.z += delta * 0.15;
        }
    });

    return (
        <group position={pos}>
            <mesh
                ref={ref}
                onClick={(e) => { e.stopPropagation(); onClick(id, pos); }}
                onPointerOver={(e) => { e.stopPropagation(); setHovered(true); onHover(id, label); }}
                onPointerOut={(e) => { e.stopPropagation(); setHovered(false); onHover(null, null); }}
            >
                <icosahedronGeometry args={[0.35, 2]} />
                <meshStandardMaterial
                    color={COLORS.gku}
                    emissive={COLORS.gku}
                    emissiveIntensity={isSelected ? 1.5 : hovered ? 1.0 : 0.5}
                    roughness={0.2}
                    metalness={0.8}
                    wireframe={false}
                />
            </mesh>
            <mesh ref={glowRef}>
                <icosahedronGeometry args={[0.48, 1]} />
                <meshBasicMaterial color={COLORS.gku} wireframe transparent opacity={0.15} />
            </mesh>
        </group>
    );
}

/* ─────────── Edges as lines ─────────── */
function GraphEdges({ edges, nodeMap }) {
    const lines = useMemo(() => {
        return edges
            .filter(e => nodeMap[e.source] && nodeMap[e.target])
            .map((e, i) => {
                const from = nodeMap[e.source];
                const to = nodeMap[e.target];
                const color = COLORS[e.type] || COLORS.semantic;
                return { from, to, color, weight: e.weight, type: e.type, key: i };
            });
    }, [edges, nodeMap]);

    return (
        <>
            {lines.map(l => (
                <Line
                    key={l.key}
                    points={[l.from, l.to]}
                    color={l.color}
                    lineWidth={l.type === 'membership' ? 1.5 : Math.max(0.5, l.weight * 2)}
                    transparent
                    opacity={l.type === 'membership' ? 0.25 : Math.max(0.08, l.weight * 0.5)}
                    dashed={l.type === 'causal'}
                    dashScale={l.type === 'causal' ? 10 : undefined}
                    dashSize={l.type === 'causal' ? 0.2 : undefined}
                    gapSize={l.type === 'causal' ? 0.1 : undefined}
                />
            ))}
        </>
    );
}

/* ─────────── Ambient floating particles ─────────── */
function FloatingParticles({ count = 200 }) {
    const ref = useRef();
    const positions = useMemo(() => {
        const pos = new Float32Array(count * 3);
        for (let i = 0; i < count * 3; i++) {
            pos[i] = (Math.random() - 0.5) * 30;
        }
        return pos;
    }, [count]);

    useFrame((_, delta) => {
        if (ref.current) ref.current.rotation.y += delta * 0.01;
    });

    return (
        <points ref={ref}>
            <bufferGeometry>
                <bufferAttribute attach="attributes-position" array={positions} count={count} itemSize={3} />
            </bufferGeometry>
            <pointsMaterial size={0.03} color="#4f8fff" transparent opacity={0.4} sizeAttenuation />
        </points>
    );
}

/* ─────────── Cinematic camera controller ─────────── */
function CameraController({ target, controlsRef }) {
    const { camera } = useThree();
    const flyTarget = useRef(null);
    const flyProgress = useRef(1);
    const startPos = useRef(new THREE.Vector3());
    const startLookAt = useRef(new THREE.Vector3());

    useEffect(() => {
        if (target) {
            startPos.current.copy(camera.position);
            if (controlsRef.current) startLookAt.current.copy(controlsRef.current.target);
            flyTarget.current = new THREE.Vector3(target[0] + 2, target[1] + 1.5, target[2] + 2);
            flyProgress.current = 0;
        }
    }, [target, camera, controlsRef]);

    useFrame((_, delta) => {
        if (flyTarget.current && flyProgress.current < 1) {
            flyProgress.current = Math.min(1, flyProgress.current + delta * 0.8);
            const t = 1 - Math.pow(1 - flyProgress.current, 3);
            camera.position.lerpVectors(startPos.current, flyTarget.current, t);
            if (controlsRef.current && target) {
                const targetVec = new THREE.Vector3(target[0], target[1], target[2]);
                controlsRef.current.target.lerpVectors(startLookAt.current, targetVec, t);
                controlsRef.current.update();
            }
        }
    });

    return null;
}


/* ═══════════════════════════════════════════════════
   Node Detail / Properties Panel (HTML overlay)
   ═══════════════════════════════════════════════════ */
function NodeDetailPanel({ node, edges, allNodes, onClose }) {
    if (!node) return null;

    const isGKU = node.type === 'gku';
    const accentColor = isGKU ? COLOR_HEX.gku : COLOR_HEX.aku;

    // Find connected edges
    const connected = useMemo(() => {
        return edges.filter(e => e.source === node.id || e.target === node.id);
    }, [edges, node.id]);

    // Build neighbor list with labels
    const nodeById = useMemo(() => {
        const m = {};
        allNodes.forEach(n => { m[n.id] = n; });
        return m;
    }, [allNodes]);

    const neighbors = useMemo(() => {
        return connected.map(e => {
            const otherId = e.source === node.id ? e.target : e.source;
            const other = nodeById[otherId];
            return {
                id: otherId,
                label: other?.label || otherId.slice(0, 8),
                type: other?.type || 'unknown',
                edgeType: e.type,
                weight: e.weight,
                semantic: e.semantic,
                causal: e.causal,
                temporal: e.temporal,
            };
        });
    }, [connected, node.id, nodeById]);

    // Group edges by type
    const edgeTypeCounts = useMemo(() => {
        const counts = {};
        connected.forEach(e => { counts[e.type] = (counts[e.type] || 0) + 1; });
        return counts;
    }, [connected]);

    return (
        <div style={{
            position: 'absolute', top: 0, right: 0, bottom: 0, width: 340,
            background: 'rgba(8,8,12,0.92)',
            backdropFilter: 'blur(24px)',
            borderLeft: '1px solid rgba(255,255,255,0.06)',
            display: 'flex', flexDirection: 'column',
            fontFamily: "'Outfit', sans-serif",
            zIndex: 30, overflow: 'hidden',
            animation: 'slideInRight 0.25s ease-out',
        }}>
            {/* Header */}
            <div style={{
                padding: '16px 16px 12px', borderBottom: '1px solid rgba(255,255,255,0.06)',
                display: 'flex', alignItems: 'flex-start', gap: 10,
            }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                        <div style={{
                            width: 10, height: 10, borderRadius: isGKU ? 3 : '50%',
                            background: accentColor, boxShadow: `0 0 8px ${accentColor}60`,
                        }} />
                        <span style={{
                            fontSize: 10, fontWeight: 700, textTransform: 'uppercase',
                            letterSpacing: '0.06em', color: accentColor,
                        }}>
                            {isGKU ? 'Generalized Knowledge Unit' : 'Atomic Knowledge Unit'}
                        </span>
                    </div>
                    <div style={{
                        fontSize: 14, fontWeight: 600, color: '#f1f5f9', lineHeight: 1.4,
                        wordBreak: 'break-word',
                    }}>
                        {isGKU ? (node.label || 'Unnamed Concept') : (node.label || 'No content')}
                    </div>
                </div>
                <button
                    onClick={onClose}
                    style={{
                        background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)',
                        borderRadius: 8, width: 28, height: 28, display: 'flex',
                        alignItems: 'center', justifyContent: 'center',
                        color: '#94a3b8', cursor: 'pointer', fontSize: 14,
                        flexShrink: 0,
                    }}
                >
                    ✕
                </button>
            </div>

            {/* Properties */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '0' }}>
                {/* Properties Section */}
                <div style={{ padding: '14px 16px', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                    <div style={sectionTitle}>Properties</div>
                    <div style={propsGrid}>
                        <PropRow label="ID" value={node.id.slice(0, 12) + '…'} mono />
                        <PropRow label="Type" value={isGKU ? 'GKU' : 'AKU'} badge badgeColor={accentColor} />
                        <PropRow label="Position" value={`(${node.x}, ${node.y}, ${node.z})`} mono />
                        {!isGKU && <PropRow label="Archived" value={node.archived ? 'Yes' : 'No'} badge badgeColor={node.archived ? '#ef4444' : '#22c55e'} />}
                        {isGKU && node.member_count != null && <PropRow label="Members" value={String(node.member_count)} />}
                    </div>
                </div>

                {/* Content Section (full text from API) */}
                {(node.content || node.label) && (
                    <div style={{ padding: '14px 16px', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                        <div style={sectionTitle}>Content</div>
                        <div style={{
                            fontSize: 12, color: '#cbd5e1', lineHeight: 1.65,
                            background: 'rgba(255,255,255,0.02)',
                            border: '1px solid rgba(255,255,255,0.04)',
                            borderRadius: 8, padding: '10px 12px',
                            maxHeight: 220, overflowY: 'auto',
                            wordBreak: 'break-word', whiteSpace: 'pre-wrap',
                        }}>
                            {node.content || node.label}
                        </div>
                    </div>
                )}

                {/* Pattern Signature (GKU only) */}
                {isGKU && node.pattern && (
                    <div style={{ padding: '14px 16px', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                        <div style={sectionTitle}>Pattern Signature</div>
                        <div style={{
                            fontSize: 11, color: '#94a3b8', fontFamily: "'JetBrains Mono', monospace",
                            background: 'rgba(255,255,255,0.02)',
                            border: '1px solid rgba(255,255,255,0.04)',
                            borderRadius: 8, padding: '10px 12px',
                            maxHeight: 120, overflowY: 'auto',
                            wordBreak: 'break-all', whiteSpace: 'pre-wrap',
                        }}>
                            {typeof node.pattern === 'object' ? JSON.stringify(node.pattern, null, 2) : String(node.pattern)}
                        </div>
                    </div>
                )}

                {/* Connections Overview */}
                <div style={{ padding: '14px 16px', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                    <div style={sectionTitle}>
                        Connections <span style={{ color: '#475569', fontWeight: 400 }}>({connected.length})</span>
                    </div>
                    {/* Edge type breakdown */}
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 10 }}>
                        {Object.entries(edgeTypeCounts).map(([type, count]) => (
                            <span key={type} style={{
                                display: 'inline-flex', alignItems: 'center', gap: 4,
                                padding: '3px 8px', borderRadius: 20, fontSize: 10, fontWeight: 600,
                                background: `${COLOR_HEX[type] || '#60a5fa'}15`,
                                border: `1px solid ${COLOR_HEX[type] || '#60a5fa'}30`,
                                color: COLOR_HEX[type] || '#60a5fa',
                            }}>
                                <span style={{
                                    width: 6, height: 6, borderRadius: '50%',
                                    background: COLOR_HEX[type] || '#60a5fa',
                                }} />
                                {type} ({count})
                            </span>
                        ))}
                    </div>

                    {/* Neighbor list */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                        {neighbors.slice(0, 20).map((nb, i) => (
                            <div key={i} style={{
                                display: 'flex', alignItems: 'center', gap: 8,
                                padding: '6px 10px', borderRadius: 8,
                                background: 'rgba(255,255,255,0.02)',
                                border: '1px solid rgba(255,255,255,0.03)',
                                fontSize: 11,
                            }}>
                                <div style={{
                                    width: 7, height: 7, borderRadius: nb.type === 'gku' ? 2 : '50%',
                                    background: nb.type === 'gku' ? COLOR_HEX.gku : COLOR_HEX.aku,
                                    flexShrink: 0,
                                }} />
                                <div style={{ flex: 1, minWidth: 0, color: '#cbd5e1', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                    {nb.label}
                                </div>
                                <span style={{
                                    fontSize: 9, padding: '1px 5px', borderRadius: 10,
                                    background: `${COLOR_HEX[nb.edgeType] || '#60a5fa'}15`,
                                    color: COLOR_HEX[nb.edgeType] || '#60a5fa',
                                    flexShrink: 0,
                                }}>
                                    {nb.edgeType}
                                </span>
                                <span style={{ fontSize: 9, color: '#475569', fontFamily: 'monospace', flexShrink: 0 }}>
                                    {nb.weight.toFixed(2)}
                                </span>
                            </div>
                        ))}
                        {neighbors.length > 20 && (
                            <div style={{ fontSize: 10, color: '#475569', padding: '4px 10px' }}>
                                +{neighbors.length - 20} more connections…
                            </div>
                        )}
                        {neighbors.length === 0 && (
                            <div style={{ fontSize: 11, color: '#475569', padding: '4px 0' }}>No connections</div>
                        )}
                    </div>
                </div>

                {/* Edge Weight Details (if AKU with semantic/causal/temporal) */}
                {connected.length > 0 && connected.some(e => e.semantic !== undefined) && (
                    <div style={{ padding: '14px 16px' }}>
                        <div style={sectionTitle}>Strongest Edges</div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                            {neighbors
                                .filter(nb => nb.semantic !== undefined)
                                .sort((a, b) => Math.max(b.semantic || 0, b.causal || 0, b.temporal || 0) - Math.max(a.semantic || 0, a.causal || 0, a.temporal || 0))
                                .slice(0, 5)
                                .map((nb, i) => (
                                    <div key={i} style={{
                                        padding: '8px 10px', borderRadius: 8,
                                        background: 'rgba(255,255,255,0.02)',
                                        border: '1px solid rgba(255,255,255,0.03)',
                                    }}>
                                        <div style={{ fontSize: 11, color: '#cbd5e1', marginBottom: 6, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                            {nb.label}
                                        </div>
                                        <div style={{ display: 'flex', gap: 12 }}>
                                            {nb.semantic != null && <WeightBar label="Sem" value={nb.semantic} color={COLOR_HEX.semantic} />}
                                            {nb.causal != null && <WeightBar label="Cau" value={nb.causal} color={COLOR_HEX.causal} />}
                                            {nb.temporal != null && <WeightBar label="Tmp" value={nb.temporal} color={COLOR_HEX.temporal} />}
                                        </div>
                                    </div>
                                ))
                            }
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

/* ─── Small helpers for the detail panel ─── */
const sectionTitle = {
    fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em',
    color: '#64748b', marginBottom: 8,
};
const propsGrid = { display: 'flex', flexDirection: 'column', gap: 6 };

function PropRow({ label, value, mono, badge, badgeColor }) {
    return (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: 11, color: '#64748b' }}>{label}</span>
            {badge ? (
                <span style={{
                    fontSize: 10, fontWeight: 600, padding: '2px 8px', borderRadius: 20,
                    background: `${badgeColor}15`, border: `1px solid ${badgeColor}30`, color: badgeColor,
                }}>
                    {value}
                </span>
            ) : (
                <span style={{
                    fontSize: 11, color: '#cbd5e1',
                    fontFamily: mono ? "'JetBrains Mono', monospace" : 'inherit',
                }}>
                    {value}
                </span>
            )}
        </div>
    );
}

function WeightBar({ label, value, color }) {
    return (
        <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                <span style={{ fontSize: 9, color: '#64748b' }}>{label}</span>
                <span style={{ fontSize: 9, color, fontFamily: 'monospace' }}>{value.toFixed(3)}</span>
            </div>
            <div style={{ height: 3, borderRadius: 2, background: 'rgba(255,255,255,0.05)' }}>
                <div style={{ height: '100%', borderRadius: 2, background: color, width: `${Math.min(100, value * 100)}%`, transition: 'width 0.3s' }} />
            </div>
        </div>
    );
}

/* ─────────── HUD overlay (stats + legend) ─────────── */
function HUD({ stats, hoveredLabel }) {
    return (
        <div style={{
            position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
            pointerEvents: 'none', fontFamily: "'Outfit', sans-serif",
        }}>
            {/* Stats panel */}
            <div style={{
                position: 'absolute', top: 16, left: 16,
                background: 'rgba(10,10,14,0.75)',
                backdropFilter: 'blur(12px)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: 10, padding: '12px 16px',
                color: '#e2e8f0', fontSize: 12, lineHeight: 1.7,
            }}>
                <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4, color: '#93c5fd' }}>Knowledge Graph</div>
                <div><span style={{ color: '#3b82f6' }}>●</span> AKUs: {stats.aku_count}</div>
                <div><span style={{ color: '#fbbf24' }}>●</span> GKUs: {stats.gku_count}</div>
                <div><span style={{ color: '#60a5fa' }}>●</span> Edges: {stats.edge_count}</div>
            </div>

            {/* Legend */}
            <div style={{
                position: 'absolute', bottom: 16, left: 16,
                background: 'rgba(10,10,14,0.75)',
                backdropFilter: 'blur(12px)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: 10, padding: '10px 14px',
                color: '#94a3b8', fontSize: 11, lineHeight: 1.8,
            }}>
                {[
                    ['#60a5fa', '── Semantic'],
                    ['#f97316', '┈┈ Causal'],
                    ['#34d399', '── Temporal'],
                    ['#a78bfa', '── Membership'],
                ].map(([c, l]) => (
                    <div key={l}><span style={{ color: c }}>━</span> {l}</div>
                ))}
            </div>

            {/* Tooltip */}
            {hoveredLabel && (
                <div style={{
                    position: 'absolute', bottom: 16, right: 16,
                    background: 'rgba(10,10,14,0.85)',
                    backdropFilter: 'blur(16px)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: 10, padding: '10px 14px',
                    color: '#f1f5f9', fontSize: 12, maxWidth: 280, wordBreak: 'break-word',
                }}>
                    {hoveredLabel}
                </div>
            )}

            {/* Controls hint */}
            <div style={{
                position: 'absolute', top: 16, right: 16,
                color: 'rgba(255,255,255,0.25)', fontSize: 11,
                textAlign: 'right', lineHeight: 1.6,
            }}>
                Scroll to zoom · Drag to orbit<br />
                Click node to inspect
            </div>
        </div>
    );
}


/* ─────────── Main Scene content ─────────── */
function SceneContent({ data, selectedNodeId, onSelectNode }) {
    const controlsRef = useRef();
    const [flyTo, setFlyTo] = useState(null);
    const [hoveredId, setHoveredId] = useState(null);
    const [hoveredLabel, setHoveredLabel] = useState(null);

    const nodeMap = useMemo(() => {
        const map = {};
        data.nodes.forEach(n => { map[n.id] = [n.x, n.y, n.z]; });
        return map;
    }, [data.nodes]);

    const handleClick = useCallback((id, pos) => {
        onSelectNode(id);
        setFlyTo(pos);
    }, [onSelectNode]);

    const handleHover = useCallback((id, label) => {
        setHoveredId(id);
        setHoveredLabel(label);
    }, []);

    const akuNodes = useMemo(() => data.nodes.filter(n => n.type === 'aku'), [data.nodes]);
    const gkuNodes = useMemo(() => data.nodes.filter(n => n.type === 'gku'), [data.nodes]);

    return (
        <>
            <ambientLight intensity={0.15} />
            <pointLight position={[10, 10, 10]} intensity={0.6} color="#a5b4fc" />
            <pointLight position={[-10, -5, -8]} intensity={0.3} color="#fbbf24" />

            <OrbitControls
                ref={controlsRef}
                enableDamping dampingFactor={0.05}
                rotateSpeed={0.5} zoomSpeed={0.8}
                minDistance={1} maxDistance={40}
            />

            <CameraController target={flyTo} controlsRef={controlsRef} />
            <Stars radius={25} depth={40} count={1500} factor={3} saturation={0.1} fade speed={0.5} />
            <FloatingParticles count={300} />

            <GraphEdges edges={data.edges} nodeMap={nodeMap} />

            {akuNodes.map(n => (
                <AKUNode key={n.id} id={n.id} pos={[n.x, n.y, n.z]}
                    label={n.label} archived={n.archived}
                    isSelected={selectedNodeId === n.id}
                    onClick={handleClick} onHover={handleHover}
                />
            ))}

            {gkuNodes.map(n => (
                <GKUNode key={n.id} id={n.id} pos={[n.x, n.y, n.z]}
                    label={n.label} pattern={n.pattern}
                    isSelected={selectedNodeId === n.id}
                    onClick={handleClick} onHover={handleHover}
                />
            ))}

            <EffectComposer>
                <Bloom luminanceThreshold={0.2} luminanceSmoothing={0.9} intensity={1.2} radius={0.8} />
            </EffectComposer>
        </>
    );
}


/* ═══════════════════════════════════════════
   MAIN EXPORT – MeshVisualization
   ═══════════════════════════════════════════ */
export default function MeshVisualization({ workspaceId }) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [stats, setStats] = useState({ aku_count: 0, gku_count: 0, edge_count: 0 });
    const [selectedNodeId, setSelectedNodeId] = useState(null);

    useEffect(() => {
        if (!workspaceId) return;
        setLoading(true);
        setError(null);
        setSelectedNodeId(null);
        fetch(`${API_BASE}/graph/${workspaceId}`)
            .then(r => { if (!r.ok) throw new Error('Failed to fetch graph'); return r.json(); })
            .then(d => { setData(d); setStats(d.stats); setLoading(false); })
            .catch(e => { setError(e.message); setLoading(false); });
    }, [workspaceId]);

    const selectedNode = useMemo(() => {
        if (!data || !selectedNodeId) return null;
        return data.nodes.find(n => n.id === selectedNodeId) || null;
    }, [data, selectedNodeId]);

    if (loading) {
        return (
            <div style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: '#64748b', fontFamily: "'Outfit', sans-serif", fontSize: 14, background: '#0a0a0c',
            }}>
                <div style={{ textAlign: 'center' }}>
                    <div style={{
                        width: 32, height: 32, border: '2px solid rgba(59,130,246,0.3)',
                        borderTop: '2px solid #3b82f6', borderRadius: '50%',
                        animation: 'spin 1s linear infinite', margin: '0 auto 12px',
                    }} />
                    Loading knowledge graph…
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: '#ef4444', fontFamily: "'Outfit', sans-serif", fontSize: 14, background: '#0a0a0c',
            }}>
                Error: {error}
            </div>
        );
    }

    if (!data || data.nodes.length === 0) {
        return (
            <div style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
                flexDirection: 'column', gap: 8,
                color: '#64748b', fontFamily: "'Outfit', sans-serif", fontSize: 14, background: '#0a0a0c',
            }}>
                <div style={{ fontSize: 32 }}>🔬</div>
                <div>No knowledge nodes yet</div>
                <div style={{ fontSize: 12, color: '#475569' }}>Train some data to populate the graph</div>
            </div>
        );
    }

    return (
        <div style={{ flex: 1, position: 'relative', background: '#050508' }}>
            <Canvas
                camera={{ position: [12, 8, 12], fov: 55, near: 0.1, far: 100 }}
                gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
                dpr={[1, 1.5]}
                style={{ background: '#050508' }}
            >
                <color attach="background" args={['#050508']} />
                <fog attach="fog" args={['#050508', 15, 40]} />
                <SceneContent data={data} selectedNodeId={selectedNodeId} onSelectNode={setSelectedNodeId} />
            </Canvas>

            <HUD stats={stats} />

            {/* Node Detail Panel */}
            <NodeDetailPanel
                node={selectedNode}
                edges={data.edges}
                allNodes={data.nodes}
                onClose={() => setSelectedNodeId(null)}
            />

            {/* CSS animation keyframes */}
            <style>{`
                @keyframes slideInRight {
                    from { transform: translateX(100%); opacity: 0; }
                    to   { transform: translateX(0);    opacity: 1; }
                }
            `}</style>
        </div>
    );
}
