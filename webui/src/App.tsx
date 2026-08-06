import { useState, useRef, useEffect } from 'react'
import { Settings2, Image as ImageIcon, Video, Square, CirclePlay, LayoutGrid, X, Download, Trash2, RefreshCcw, PlayCircle, ChevronLeft, ChevronRight, GripVertical } from 'lucide-react'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Checkbox } from '@/components/ui/checkbox'
import { Dialog, DialogContent, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

type HistoryItem = {
  filename: string;
  url: string;
  type: string;
  created_at: number;
}

function useStickyState<T>(defaultValue: T, key: string): [T, React.Dispatch<React.SetStateAction<T>>] {
  const [value, setValue] = useState<T>(() => {
    const stickyValue = window.localStorage.getItem(key);
    return stickyValue !== null ? JSON.parse(stickyValue) : defaultValue;
  });
  useEffect(() => {
    window.localStorage.setItem(key, JSON.stringify(value));
  }, [key, value]);
  return [value, setValue];
}

export default function App() {
  const [leftOpen, setLeftOpen] = useState(true)
  const [rightOpen, setRightOpen] = useState(false)
  const [previewVisible, setPreviewVisible] = useState(true)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  
  const [sourceFile, setSourceFile] = useState<File | null>(null)
  const [sourcePreview, setSourcePreview] = useState<string>('')
  
  const [targetFiles, setTargetFiles] = useState<File[]>([])
  const [targetPreview, setTargetPreview] = useState<string>('')
  const [targetType, setTargetType] = useState<string>('video')
  
  // Settings (Persistent)
  const [swapModel, setSwapModel] = useStickyState('inswapper_128', 'setting_swapModel')
  const [restoreModel, setRestoreModel] = useStickyState('gfpgan_1.4', 'setting_restoreModel')
  const [swapWeight, setSwapWeight] = useStickyState([65], 'setting_swapWeight')
  const [swapBoost, setSwapBoost] = useStickyState([128], 'setting_swapBoost')
  const [restoreWeight, setRestoreWeight] = useStickyState([100], 'setting_restoreWeight')
  const [restoreBlend, setRestoreBlend] = useStickyState([100], 'setting_restoreBlend')
  const [faceRestore, setFaceRestore] = useStickyState(true, 'setting_faceRestore')
  const [colorMatch, setColorMatch] = useStickyState(false, 'setting_colorMatch')
  const [similarity, setSimilarity] = useStickyState(false, 'setting_similarity')
  const [previewFreq, setPreviewFreq] = useStickyState([15], 'setting_previewFreq')

  const [currentJobId, setCurrentJobId] = useState<string | null>(null)
  const [ws, setWs] = useState<WebSocket | null>(null)
  
  // History State
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set())
  const [lightboxItem, setLightboxItem] = useState<HistoryItem | null>(null)

  // Floating Action Row state
  const [actionPos, setActionPos] = useState({ x: 0, y: 0 })
  const isDraggingRef = useRef(false)
  const dragStartRef = useRef({ x: 0, y: 0 })

  const API_BASE = `http://${window.location.hostname}:8000`
  const PLATFORM = "webui_react"

  const sourceInputRef = useRef<HTMLInputElement>(null)
  const targetInputRef = useRef<HTMLInputElement>(null)

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/history`, { headers: { 'X-Client-Platform': PLATFORM } })
      const data = await res.json()
      const formattedHistory = (data.history || []).map((item: any) => ({
        ...item,
        url: item.url.startsWith('http') ? item.url : `${API_BASE}${item.url}`
      }))
      setHistory(formattedHistory)
    } catch (e) {
      console.error("Failed to fetch history")
    }
  }

  useEffect(() => {
    fetchHistory()
  }, [])

  const toggleLeft = () => { setLeftOpen(!leftOpen); setRightOpen(false) }
  const toggleRight = () => { setRightOpen(!rightOpen); setLeftOpen(false) }
  const closePanels = () => { setLeftOpen(false); setRightOpen(false) }

  const handleSourceChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const f = e.target.files[0]
      setSourceFile(f)
      setSourcePreview(URL.createObjectURL(f))
    }
  }

  const handleTargetChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const files = Array.from(e.target.files)
      setTargetFiles(files)
      const file = files[0] // show preview of first file
      if (file.type.startsWith('video/')) {
        setTargetType('video')
        setTargetPreview(URL.createObjectURL(file))
      } else if (file.type.startsWith('image/')) {
        setTargetType('image')
        setTargetPreview(URL.createObjectURL(file))
      } else {
        setTargetPreview('')
      }
    }
  }

  const connectWebSocket = (jobId: string) => {
    const wsUrl = `ws://${window.location.hostname}:8000/api/v1/ws/jobs/${jobId}`
    const websocket = new WebSocket(wsUrl)
    
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setProgress(data.progress || 0)
      
      if (data.preview_image) {
        setTargetType('image')
        setTargetPreview(data.preview_image)
      }
      
      if (data.status === 'completed') {
        setRunning(false)
        if (data.output_path) {
          const filename = data.output_path.split(/[\\/]/).pop()
          setTargetPreview(`${API_BASE}/api/v1/history/${filename}?platform=${PLATFORM}`)
        }
        fetchHistory()
        websocket.close()
      } else if (data.status === 'failed') {
        setRunning(false)
        console.error("Processing Failed: " + data.error)
        websocket.close()
      }
    }
    setWs(websocket)
  }

  const uploadFile = async (file: File) => {
    const formData = new FormData()
    formData.append('files', file)
    const res = await fetch(`${API_BASE}/api/v1/upload`, {
      method: 'POST',
      headers: { 'X-Client-Platform': PLATFORM },
      body: formData
    })
    const data = await res.json()
    return data.uploaded[0].file_id
  }

  const toggleRun = async () => {
    if (running && currentJobId) {
      setRunning(false)
      setProgress(0)
      if (ws) ws.close()
      await fetch(`${API_BASE}/api/v1/jobs/${currentJobId}/cancel`, { method: 'POST' })
    } else {
      if (!sourceFile || targetFiles.length === 0) {
        alert("Please upload both Source and Target files.")
        return
      }
      setRunning(true)
      setProgress(0)
      try {
        const sourceId = await uploadFile(sourceFile)
        
        // Upload all targets in parallel
        const targetUploadPromises = targetFiles.map(f => uploadFile(f))
        const targetIds = await Promise.all(targetUploadPromises)

        let processors = ["swap"]
        if (faceRestore) processors.push("restore")
        if (colorMatch) processors.push("color")

        const jobRes = await fetch(`${API_BASE}/api/v1/jobs`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Client-Platform': PLATFORM },
          body: JSON.stringify({
            source_file_id: sourceId,
            target_file_ids: targetIds,
            preview_frequency: previewFreq[0],
            processors: processors,
            swap_model: swapModel,
            swap_weight: swapWeight[0] / 100,
            swap_boost: swapBoost[0],
            restore_model: restoreModel,
            restore_weight: restoreWeight[0] / 100,
            restore_blend: restoreBlend[0],
            similarity: similarity
          })
        })
        const jobData = await jobRes.json()
        setCurrentJobId(jobData.job_id)
        connectWebSocket(jobData.job_id)
      } catch (err) {
        alert("Error starting job: " + err)
        setRunning(false)
      }
    }
  }

  const handleSelectToggle = (filename: string, checked: boolean) => {
    const newSet = new Set(selectedItems)
    if (checked) {
      newSet.add(filename)
    } else {
      newSet.delete(filename)
    }
    setSelectedItems(newSet)
  }

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      const allFiles = history.map(item => item.filename)
      setSelectedItems(new Set(allFiles))
    } else {
      setSelectedItems(new Set())
    }
  }

  const bulkDelete = async () => {
    const files = Array.from(selectedItems)
    if (!files.length) return
    if (!window.confirm(`คุณต้องการลบผลลัพธ์จำนวน ${files.length} รายการ ใช่หรือไม่?`)) return
    
    await fetch(`${API_BASE}/api/v1/history`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json', 'X-Client-Platform': PLATFORM },
      body: JSON.stringify({ filenames: files })
    })
    setSelectedItems(new Set())
    fetchHistory()
  }

  const bulkDownload = async () => {
    const files = Array.from(selectedItems)
    if (!files.length) return
    const res = await fetch(`${API_BASE}/api/v1/history/download`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Client-Platform': PLATFORM },
      body: JSON.stringify({ filenames: files })
    })
    const blob = await res.blob()
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = files.length === 1 ? files[0] : `uni-face-export-${Date.now()}.zip`
    document.body.appendChild(a)
    a.click()
    a.remove()
    window.URL.revokeObjectURL(url)
  }

  const handleDragStart = (e: React.PointerEvent) => {
    isDraggingRef.current = true
    dragStartRef.current = { x: e.clientX - actionPos.x, y: e.clientY - actionPos.y }
    e.currentTarget.setPointerCapture(e.pointerId)
  }
  const handleDragMove = (e: React.PointerEvent) => {
    if (!isDraggingRef.current) return
    setActionPos({
      x: e.clientX - dragStartRef.current.x,
      y: e.clientY - dragStartRef.current.y
    })
  }
  const handleDragEnd = (e: React.PointerEvent) => {
    isDraggingRef.current = false
    e.currentTarget.releasePointerCapture(e.pointerId)
  }

  return (
    <div className="app">
      {/* Topbar */}
      <header className="topbar">
        <button className={`icon-btn ${leftOpen ? 'active' : ''}`} onClick={toggleLeft} title="Output Library">
          <LayoutGrid />
        </button>
        <div className="brand"><span className="brand-mark">◆</span> UNI-FACE</div>
        <button className={`icon-btn ${rightOpen ? 'active' : ''}`} onClick={toggleRight} title="Settings">
          <Settings2 />
        </button>
      </header>

      {/* Status strip */}
      <div className="status-strip">
        <span className={`status-dot ${running ? 'running' : ''}`}></span>
        <span className="status-text">{running ? 'RUNNING' : 'IDLE'}</span>
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${progress}%` }}></div>
        </div>
        <span className="progress-pct">{progress}%</span>
      </div>

      <div className={`scrim ${leftOpen || rightOpen ? 'show' : ''}`} onClick={closePanels}></div>

      <div className="workspace">
        {/* Output panel (LEFT) */}
        <aside className={`panel panel-left ${leftOpen ? 'open' : ''}`}>
          <div className="panel-header">
            OUTPUT LIBRARY
            <div className="flex gap-2">
              <button className="panel-close" onClick={fetchHistory} title="Refresh"><RefreshCcw size={14}/></button>
              <button className="panel-close" onClick={closePanels}><X size={16}/></button>
            </div>
          </div>
          <div className="panel-body">
            <div className="output-grid">
              {history.length === 0 ? (
                <div className="output-empty">ยังไม่มีผลลัพธ์<br/>กด START เพื่อเริ่มประมวลผล</div>
              ) : (
                history.map((item) => (
                  <div key={item.filename} className={`output-item ${selectedItems.has(item.filename) ? 'selected' : ''}`}>
                    <div className="output-check" onClick={(e) => e.stopPropagation()}>
                      <Checkbox 
                        checked={selectedItems.has(item.filename)} 
                        onCheckedChange={(c) => handleSelectToggle(item.filename, !!c)} 
                      />
                    </div>
                    <div className="w-full h-full relative flex items-center justify-center" onClick={() => setLightboxItem(item)}>
                      {item.type === 'video' ? (
                        <>
                          <video src={item.url} muted loop playsInline />
                          <div className="absolute inset-0 bg-black/30 flex items-center justify-center pointer-events-none">
                            <PlayCircle size={32} className="text-white opacity-80" />
                          </div>
                        </>
                      ) : (
                        <img src={item.url} alt={item.filename} loading="lazy" />
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
          
          {/* Floating Action Bar (Moved inside panel) */}
          {selectedItems.size > 0 && (
            <div className="floating-action-bar">
              <div className="flex items-center gap-2">
                <Checkbox 
                  checked={selectedItems.size === history.length} 
                  onCheckedChange={(c) => handleSelectAll(!!c)} 
                />
                <span className="floating-text">Selected: {selectedItems.size}</span>
              </div>
              <div className="floating-divider" />
              <button className="icon-btn" style={{color: 'var(--text-primary)'}} onClick={bulkDownload} title="Download">
                <Download size={18} />
              </button>
              <button className="icon-btn" style={{color: 'var(--danger)'}} onClick={bulkDelete} title="Delete">
                <Trash2 size={18} />
              </button>
            </div>
          )}
        </aside>

        {/* Stage */}
        <main className="stage">
          <div className="preview-frame">
            <div className="hud-corner tl"></div>
            <div className="hud-corner tr"></div>
            <div className="hud-corner bl"></div>
            <div className="hud-corner br"></div>
            <div className={`hud-rec ${running ? 'live' : ''}`}>
              <span className="dot"></span><span>{running ? 'REC' : 'STBY'}</span>
            </div>
            <div className="hud-time">--:--:--</div>

            {targetPreview ? (
              targetType === 'video' ? (
                <video className="main-video show" src={targetPreview} controls playsInline autoPlay muted loop></video>
              ) : (
                <img className="main-video show" src={targetPreview} alt="Target Preview" />
              )
            ) : (
              <div className="preview-placeholder">
                <Video size={46} opacity={0.5} />
                <p>อัปโหลดไฟล์ TARGET เพื่อดูตัวอย่างวิดีโอที่นี่</p>
              </div>
            )}

            {!previewVisible && (
              <div className="preview-off-overlay show">
                <Video size={30} />
                <span>PREVIEW OFF</span>
              </div>
            )}
          </div>

          <div className="input-row">
            <div className={`input-card ${sourceFile ? 'filled' : ''}`} onClick={() => sourceInputRef.current?.click()}>
              <input type="file" accept="image/*" hidden ref={sourceInputRef} onChange={handleSourceChange} />
              <div className="input-icon">
                {sourcePreview ? <img src={sourcePreview} alt="source" /> : <ImageIcon />}
              </div>
              <div className="input-text">
                <span className="input-label">SOURCE FACE</span>
                <span className="input-sub">{sourceFile ? sourceFile.name : 'คลิกเพื่ออัปโหลด · 1 ไฟล์'}</span>
              </div>
            </div>

            <div className={`input-card ${targetFiles.length > 0 ? 'filled' : ''}`} onClick={() => targetInputRef.current?.click()}>
              <input type="file" accept="image/*,video/*" multiple hidden ref={targetInputRef} onChange={handleTargetChange} />
              <div className="input-icon">
                <Video />
              </div>
              <div className="input-text">
                <span className="input-label">TARGET</span>
                <span className="input-sub">{targetFiles.length > 0 ? targetFiles.map(f => f.name).join(', ') : 'คลิกเพื่ออัปโหลด · เลือกได้หลายไฟล์'}</span>
              </div>
              {targetFiles.length > 0 && <span className="input-count show">{targetFiles.length}</span>}
            </div>
          </div>
        </main>
        
        {/* Floating Action Bar (Draggable) */}
        <div 
          className="action-row draggable-fab"
          style={{ transform: `translate(${actionPos.x}px, ${actionPos.y}px)` }}
        >
          <div 
            className="drag-handle"
            onPointerDown={handleDragStart}
            onPointerMove={handleDragMove}
            onPointerUp={handleDragEnd}
            onPointerCancel={handleDragEnd}
            title="Drag to move"
          >
            <GripVertical size={18} />
          </div>
          <button className={`btn-core btn-primary ${running ? 'running' : ''}`} onClick={toggleRun}>
            {running ? <Square fill="currentColor" /> : <CirclePlay fill="currentColor" />}
            <span>{running ? 'CANCEL' : 'START'}</span>
          </button>
          <button className={`btn-core btn-ghost ${!previewVisible ? 'off' : ''}`} onClick={() => setPreviewVisible(!previewVisible)}>
            <Video />
            <span>{previewVisible ? 'PREVIEW' : 'NO PREVIEW'}</span>
          </button>
        </div>

        {/* Settings panel (RIGHT) */}
        <aside className={`panel panel-right ${rightOpen ? 'open' : ''}`}>
          <div className="panel-header">
            SETTINGS
            <button className="panel-close" onClick={closePanels}><X size={16}/></button>
          </div>
          <div className="panel-body">
            <div className="setting-item">
              <div className="setting-label">Swap Model</div>
              <Select value={swapModel} onValueChange={setSwapModel}>
                <SelectTrigger className="w-full h-8 text-xs bg-black/40 border-slate-700">
                  <SelectValue placeholder="Select swap model" />
                </SelectTrigger>
                <SelectContent className="bg-[#151515] text-slate-200 border-slate-700">
                  <SelectItem value="inswapper_128">inswapper_128</SelectItem>
                  <SelectItem value="inswapper_128_fp16">inswapper_128_fp16</SelectItem>
                  <SelectItem value="hyperswap_1b_256">hyperswap_1b_256</SelectItem>
                  <SelectItem value="hyperswap_1c_256">hyperswap_1c_256</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="setting-item">
              <div className="setting-label">Swap Weight <span className="setting-value">{swapWeight[0]}%</span></div>
              <Slider value={swapWeight} onValueChange={setSwapWeight} max={100} step={1} />
            </div>
            <div className="setting-item">
              <div className="setting-label">Swap Boost <span className="setting-value">{swapBoost[0]}</span></div>
              <Slider value={swapBoost} onValueChange={setSwapBoost} max={256} step={1} />
            </div>
            
            <hr className="my-4 border-slate-800" />
            
            <div className="setting-item">
              <div className="setting-label">Restore Model</div>
              <Select value={restoreModel} onValueChange={setRestoreModel}>
                <SelectTrigger className="w-full h-8 text-xs bg-black/40 border-slate-700">
                  <SelectValue placeholder="Select restore model" />
                </SelectTrigger>
                <SelectContent className="bg-[#151515] text-slate-200 border-slate-700">
                  <SelectItem value="gfpgan_1.4">gfpgan_1.4</SelectItem>
                  <SelectItem value="codeformer">codeformer</SelectItem>
                  <SelectItem value="gpen_bfr_256">gpen_bfr_256</SelectItem>
                  <SelectItem value="gpen_bfr_512">gpen_bfr_512</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="setting-item">
              <div className="setting-label">Restore Weight <span className="setting-value">{restoreWeight[0]}%</span></div>
              <Slider value={restoreWeight} onValueChange={setRestoreWeight} max={100} step={1} />
            </div>
            <div className="setting-item">
              <div className="setting-label">Restore Blend <span className="setting-value">{restoreBlend[0]}%</span></div>
              <Slider value={restoreBlend} onValueChange={setRestoreBlend} max={100} step={1} />
            </div>
            
            <hr className="my-4 border-slate-800" />
            
            <div className="setting-item toggle-item pt-2">
              <div className="setting-label !mb-0">Face Restore</div>
              <Switch checked={faceRestore} onCheckedChange={setFaceRestore} />
            </div>
            <div className="setting-item toggle-item">
              <div className="setting-label !mb-0">Color Match</div>
              <Switch checked={colorMatch} onCheckedChange={setColorMatch} />
            </div>
            <div className="setting-item toggle-item">
              <div className="setting-label !mb-0">Similarity Check</div>
              <Switch checked={similarity} onCheckedChange={setSimilarity} />
            </div>

            <hr className="my-4 border-slate-800" />

            <div className="setting-item">
              <div className="setting-label">Preview Frame Every <span className="setting-value">{previewFreq[0]}</span></div>
              <Slider value={previewFreq} onValueChange={setPreviewFreq} max={60} min={1} step={1} />
            </div>
            
            <p className="panel-note">การตั้งค่าถูกบันทึกอัตโนมัติ</p>
          </div>
        </aside>
      </div>

      {/* Lightbox Modal */}
      <Dialog open={!!lightboxItem} onOpenChange={(o) => !o && setLightboxItem(null)}>
        <DialogContent className="max-w-4xl p-0 overflow-hidden border-none bg-black/90">
          <DialogTitle className="hidden">Preview</DialogTitle>
          <DialogDescription className="hidden">Lightbox preview of output</DialogDescription>
          {lightboxItem && (() => {
            const currentIndex = history.findIndex(h => h.filename === lightboxItem.filename)
            const hasPrev = currentIndex > 0
            const hasNext = currentIndex < history.length - 1

            return (
              <div className="flex items-center justify-center w-full h-[80vh] relative group">
                {/* Prev Button */}
                {hasPrev && (
                  <button 
                    className="absolute left-4 z-50 p-2 rounded-full bg-black/50 text-white hover:bg-black/80 transition opacity-0 group-hover:opacity-100"
                    onClick={(e) => { e.stopPropagation(); setLightboxItem(history[currentIndex - 1]) }}
                  >
                    <ChevronLeft size={32} />
                  </button>
                )}
                
                {/* Next Button */}
                {hasNext && (
                  <button 
                    className="absolute right-4 z-50 p-2 rounded-full bg-black/50 text-white hover:bg-black/80 transition opacity-0 group-hover:opacity-100"
                    onClick={(e) => { e.stopPropagation(); setLightboxItem(history[currentIndex + 1]) }}
                  >
                    <ChevronRight size={32} />
                  </button>
                )}

                {lightboxItem.type === 'video' ? (
                  <video src={lightboxItem.url} controls autoPlay loop className="max-w-full max-h-full object-contain" />
                ) : (
                  <img src={lightboxItem.url} alt="Preview" className="max-w-full max-h-full object-contain" />
                )}
              </div>
            )
          })()}
        </DialogContent>
      </Dialog>
    </div>
  )
}
