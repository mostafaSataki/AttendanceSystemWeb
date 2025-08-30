'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Badge } from '@/components/ui/badge'
import { Checkbox } from '@/components/ui/checkbox'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Textarea } from '@/components/ui/textarea'

export default function FaceRecognitionSystem() {
  const [mainTab, setMainTab] = useState('recognition')
  const [recognitionSource, setRecognitionSource] = useState('video')
  const [videoFilePath, setVideoFilePath] = useState('')
  const [rtspUrl, setRtspUrl] = useState('')
  const [tracker, setTracker] = useState('bytetrack')
  const [confidence, setConfidence] = useState([70])
  const [showDetectionBoxes, setShowDetectionBoxes] = useState(true)
  const [showTrackTrajectories, setShowTrackTrajectories] = useState(true)
  const [showRecognitionNames, setShowRecognitionNames] = useState(true)
  const [showMonitoringRegions, setShowMonitoringRegions] = useState(false)
  const [autoScroll, setAutoScroll] = useState(true)
  const [showSettingsDialog, setShowSettingsDialog] = useState(false)
  const [showSourceDialog, setShowSourceDialog] = useState(false)
  const [activeTrafficTab, setActiveTrafficTab] = useState('current')
  const [showGateDialog, setShowGateDialog] = useState(false)
  const [editingGate, setEditingGate] = useState(null)
  const [gates, setGates] = useState([
    {
      id: 1,
      name: "Main Entrance",
      description: "Main building entrance gate",
      inputCamera: {
        id: 1,
        name: "Entrance Camera 1",
        rtspUrl: "rtsp://192.168.1.100:554/entrance",
        status: "active"
      },
      outputCamera: {
        id: 2,
        name: "Exit Camera 1", 
        rtspUrl: "rtsp://192.168.1.101:554/exit",
        status: "active"
      },
      status: "active",
      createdAt: "2024-01-10"
    },
    {
      id: 2,
      name: "Parking Gate",
      description: "Parking area entrance and exit",
      inputCamera: {
        id: 3,
        name: "Parking Entry Camera",
        rtspUrl: "rtsp://192.168.1.102:554/parking-entry",
        status: "active"
      },
      outputCamera: {
        id: 4,
        name: "Parking Exit Camera",
        rtspUrl: "rtsp://192.168.1.103:554/parking-exit", 
        status: "inactive"
      },
      status: "active",
      createdAt: "2024-01-12"
    }
  ])
  const [gateForm, setGateForm] = useState({
    name: '',
    description: '',
    inputCameraName: '',
    inputCameraRtsp: '',
    outputCameraName: '',
    outputCameraRtsp: ''
  })
  const [trafficReports, setTrafficReports] = useState([
    {
      id: 1,
      personName: "John Doe",
      gateName: "Main Entrance",
      direction: "entry",
      timestamp: "2024-01-15 14:30:25",
      confidence: 95,
      cameraName: "Entrance Camera 1",
      status: "authorized",
      imageUrl: "/api/placeholder/64/64"
    },
    {
      id: 2,
      personName: "Jane Smith",
      gateName: "Main Entrance", 
      direction: "exit",
      timestamp: "2024-01-15 14:28:12",
      confidence: 87,
      cameraName: "Exit Camera 1",
      status: "authorized",
      imageUrl: "/api/placeholder/64/64"
    },
    {
      id: 3,
      personName: "Ahmed Hassan",
      gateName: "Parking Gate",
      direction: "entry",
      timestamp: "2024-01-15 14:25:45",
      confidence: 92,
      cameraName: "Parking Entry Camera",
      status: "authorized",
      imageUrl: "/api/placeholder/64/64"
    },
    {
      id: 4,
      personName: "Maria Garcia",
      gateName: "Parking Gate",
      direction: "exit", 
      timestamp: "2024-01-15 14:22:33",
      confidence: 89,
      cameraName: "Parking Exit Camera",
      status: "unauthorized",
      imageUrl: "/api/placeholder/64/64"
    },
    {
      id: 5,
      personName: "Robert Johnson",
      gateName: "Main Entrance",
      direction: "entry",
      timestamp: "2024-01-15 14:20:15",
      confidence: 94,
      cameraName: "Entrance Camera 1",
      status: "authorized",
      imageUrl: "/api/placeholder/64/64"
    }
  ])
  const [trafficFilters, setTrafficFilters] = useState({
    gate: 'all',
    direction: 'all',
    status: 'all',
    dateFrom: '',
    dateTo: '',
    search: ''
  })
  const [enrollmentSource, setEnrollmentSource] = useState('video')
  const [isEnrolling, setIsEnrolling] = useState(false)
  const [enrollmentStatus, setEnrollmentStatus] = useState('Ready')
  const [isRecognizing, setIsRecognizing] = useState(false)
  const [detectionResults, setDetectionResults] = useState<string[]>([])
  const [transactions, setTransactions] = useState([
    {
      id: 1,
      name: "John Doe",
      time: "14:30:25",
      date: "2024-01-15",
      confidence: 95,
      image: "/api/placeholder/64/64"
    },
    {
      id: 2,
      name: "Jane Smith", 
      time: "14:28:12",
      date: "2024-01-15",
      confidence: 87,
      image: "/api/placeholder/64/64"
    },
    {
      id: 3,
      name: "Ahmed Hassan",
      time: "14:25:45", 
      date: "2024-01-15",
      confidence: 92,
      image: "/api/placeholder/64/64"
    },
    {
      id: 4,
      name: "Maria Garcia",
      time: "14:22:33",
      date: "2024-01-15", 
      confidence: 89,
      image: "/api/placeholder/64/64"
    }
  ])

  const handleStartRecognition = () => {
    setIsRecognizing(true)
    // Simulate detection results and add new transaction
    const newTransaction = {
      id: transactions.length + 1,
      name: "New Person " + (transactions.length + 1),
      time: new Date().toLocaleTimeString('en-US', { hour12: false }),
      date: new Date().toISOString().split('T')[0],
      confidence: Math.floor(Math.random() * 15) + 85, // Random confidence between 85-99%
      image: "/api/placeholder/64/64"
    }
    setTransactions([newTransaction, ...transactions])
    setDetectionResults(['Person detected - ' + newTransaction.confidence + '% confidence'])
  }

  const handleClearDisplay = () => {
    setIsRecognizing(false)
    setDetectionResults([])
  }

  const handleClearResults = () => {
    setDetectionResults([])
    setTransactions([])
  }

  const handleSaveResults = () => {
    // Simulate saving results
    console.log('Saving results:', detectionResults)
  }

  // Gate CRUD Functions
  const handleAddGate = () => {
    setEditingGate(null)
    setGateForm({
      name: '',
      description: '',
      inputCameraName: '',
      inputCameraRtsp: '',
      outputCameraName: '',
      outputCameraRtsp: ''
    })
    setShowGateDialog(true)
  }

  const handleEditGate = (gate) => {
    setEditingGate(gate)
    setGateForm({
      name: gate.name,
      description: gate.description,
      inputCameraName: gate.inputCamera.name,
      inputCameraRtsp: gate.inputCamera.rtspUrl,
      outputCameraName: gate.outputCamera.name,
      outputCameraRtsp: gate.outputCamera.rtspUrl
    })
    setShowGateDialog(true)
  }

  const handleSaveGate = () => {
    if (editingGate) {
      // Update existing gate
      const updatedGates = gates.map(gate => 
        gate.id === editingGate.id 
          ? {
              ...gate,
              name: gateForm.name,
              description: gateForm.description,
              inputCamera: {
                ...gate.inputCamera,
                name: gateForm.inputCameraName,
                rtspUrl: gateForm.inputCameraRtsp
              },
              outputCamera: {
                ...gate.outputCamera,
                name: gateForm.outputCameraName,
                rtspUrl: gateForm.outputCameraRtsp
              }
            }
          : gate
      )
      setGates(updatedGates)
    } else {
      // Add new gate
      const newGate = {
        id: Math.max(...gates.map(g => g.id)) + 1,
        name: gateForm.name,
        description: gateForm.description,
        inputCamera: {
          id: Math.max(...gates.flatMap(g => [g.inputCamera.id, g.outputCamera.id])) + 1,
          name: gateForm.inputCameraName,
          rtspUrl: gateForm.inputCameraRtsp,
          status: "active"
        },
        outputCamera: {
          id: Math.max(...gates.flatMap(g => [g.inputCamera.id, g.outputCamera.id])) + 2,
          name: gateForm.outputCameraName,
          rtspUrl: gateForm.outputCameraRtsp,
          status: "active"
        },
        status: "active",
        createdAt: new Date().toISOString().split('T')[0]
      }
      setGates([newGate, ...gates])
    }
    setShowGateDialog(false)
  }

  const handleDeleteGate = (gateId) => {
    if (confirm('Are you sure you want to delete this gate?')) {
      setGates(gates.filter(gate => gate.id !== gateId))
    }
  }

  const handleToggleGateStatus = (gateId) => {
    setGates(gates.map(gate => 
      gate.id === gateId 
        ? { ...gate, status: gate.status === 'active' ? 'inactive' : 'active' }
        : gate
    ))
  }

  const handleStartEnrollment = () => {
    setIsEnrolling(true)
    setEnrollmentStatus('Enrolling...')
    // Simulate enrollment process
  }

  const handleClearEnrollmentDisplay = () => {
    setIsEnrolling(false)
    setEnrollmentStatus('Ready')
  }

  return (
    <div className="flex h-screen bg-background">
      <Tabs value={mainTab} onValueChange={setMainTab} className="flex-1 flex flex-col h-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="recognition">Face Recognition</TabsTrigger>
          <TabsTrigger value="enrollment">Face Enrollment</TabsTrigger>
          <TabsTrigger value="gates">Gates & Cameras</TabsTrigger>
        </TabsList>
        
        <TabsContent value="recognition" className="flex-1 flex m-0 p-0 data-[state=active]:flex h-full">
          {/* Main Recognition Interface */}
          <div className="flex h-full w-full min-h-0">
            {/* Left Sidebar */}
            <div className="w-80 bg-card border-r p-4 flex flex-col gap-4 flex-shrink-0">
              {/* Recognition Source Button */}
              <Button onClick={() => setShowSourceDialog(true)} className="w-full" variant="outline">
                Recognition Source
              </Button>

              {/* Recognition Settings Button */}
              <Button onClick={() => setShowSettingsDialog(true)} className="w-full" variant="outline">
                Recognition Settings
              </Button>

              {/* Recognition Controls */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Recognition Controls</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <Button 
                    onClick={handleStartRecognition} 
                    className="w-full" 
                    disabled={isRecognizing}
                  >
                    Start Recognition
                  </Button>
                  <Button 
                    onClick={handleClearDisplay} 
                    variant="outline" 
                    className="w-full"
                  >
                    Clear Display
                  </Button>
                </CardContent>
              </Card>

              {/* Current Traffic */}
              <Card className="flex-1">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Current Traffic</CardTitle>
                  <Tabs value={activeTrafficTab} onValueChange={setActiveTrafficTab} className="w-full">
                    <TabsList className="grid w-full grid-cols-2 h-8">
                      <TabsTrigger value="current" className="text-xs">Current</TabsTrigger>
                      <TabsTrigger value="traffic" className="text-xs">Traffic</TabsTrigger>
                    </TabsList>
                  </Tabs>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Tabs value={activeTrafficTab} className="w-full">
                    <TabsContent value="current" className="m-0 p-0">
                      <ScrollArea className="h-64">
                        <div className="space-y-3">
                          {transactions.slice(0, 2).map((transaction) => (
                            <Card key={transaction.id} className="p-3">
                              <div className="flex items-center space-x-3">
                                <Avatar className="w-12 h-12">
                                  <AvatarImage src={transaction.image} alt={transaction.name} />
                                  <AvatarFallback>
                                    {transaction.name.split(' ').map(n => n[0]).join('')}
                                  </AvatarFallback>
                                </Avatar>
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center justify-between">
                                    <p className="text-sm font-medium truncate">{transaction.name}</p>
                                    <Badge variant="secondary" className="text-xs">
                                      {transaction.confidence}%
                                    </Badge>
                                  </div>
                                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                                    <span>{transaction.date}</span>
                                    <span>{transaction.time}</span>
                                  </div>
                                </div>
                              </div>
                            </Card>
                          ))}
                        </div>
                      </ScrollArea>
                    </TabsContent>
                    
                    <TabsContent value="traffic" className="m-0 p-0">
                      <ScrollArea className="h-64">
                        <div className="space-y-3">
                          {transactions.map((transaction) => (
                            <Card key={transaction.id} className="p-3">
                              <div className="flex items-center space-x-3">
                                <Avatar className="w-12 h-12">
                                  <AvatarImage src={transaction.image} alt={transaction.name} />
                                  <AvatarFallback>
                                    {transaction.name.split(' ').map(n => n[0]).join('')}
                                  </AvatarFallback>
                                </Avatar>
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center justify-between">
                                    <p className="text-sm font-medium truncate">{transaction.name}</p>
                                    <Badge variant="secondary" className="text-xs">
                                      {transaction.confidence}%
                                    </Badge>
                                  </div>
                                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                                    <span>{transaction.date}</span>
                                    <span>{transaction.time}</span>
                                  </div>
                                </div>
                              </div>
                            </Card>
                          ))}
                        </div>
                      </ScrollArea>
                    </TabsContent>
                  </Tabs>
                  
                  <div className="flex gap-2">
                    <Button onClick={handleClearResults} variant="outline" className="flex-1">
                      Clear Results
                    </Button>
                    <Button onClick={handleSaveResults} variant="outline" className="flex-1">
                      Save Results
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Main Content Area */}
            <div className="flex-1 flex flex-col min-w-0">
              {/* Video Display Area */}
              <div className="flex-1 bg-black flex items-center justify-center relative">
                <div className="text-center text-white">
                  <div className="text-lg mb-2">No Video - Select source and start recognition</div>
                  <div className="text-sm text-gray-400">No video loaded</div>
                </div>
              </div>

              {/* Status Bar */}
              <div className="h-8 bg-muted border-t flex items-center px-4">
                <span className="text-sm text-muted-foreground">No video loaded</span>
              </div>
            </div>
          </div>
        </TabsContent>
        
        <TabsContent value="enrollment" className="flex-1 flex m-0 p-0 data-[state=active]:flex h-full">
          {/* Face Enrollment Interface */}
          <div className="flex h-full w-full min-h-0">
            {/* Left Panel - Controls and List */}
            <div className="w-96 bg-card border-r p-6 flex flex-col gap-6 flex-shrink-0">
              {/* Enrollment Source */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Enrollment Source</CardTitle>
                </CardHeader>
                <CardContent>
                  <RadioGroup value={enrollmentSource} onValueChange={setEnrollmentSource}>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="video" id="enrollment-video" />
                      <Label htmlFor="enrollment-video">Video File</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="camera" id="enrollment-camera" />
                      <Label htmlFor="enrollment-camera">IP Camera</Label>
                    </div>
                  </RadioGroup>
                </CardContent>
              </Card>

              {/* Enrollment Controls */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Enrollment Controls</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <Button 
                    onClick={handleStartEnrollment} 
                    className="w-full" 
                    disabled={isEnrolling}
                  >
                    Start Enrollment
                  </Button>
                  <Button 
                    onClick={handleClearEnrollmentDisplay} 
                    variant="outline" 
                    className="w-full"
                  >
                    Clear Display
                  </Button>
                </CardContent>
              </Card>

              {/* Status */}
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center">
                    <span className="text-sm font-medium">{enrollmentStatus}</span>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Right Panel - Video Display */}
            <div className="flex-1 flex flex-col min-w-0">
              {/* Video Display Area */}
              <div className="flex-1 bg-black flex items-center justify-center relative">
                <div className="text-center text-white">
                  <div className="text-lg mb-2">No Image</div>
                  <div className="text-sm text-gray-400">No video loaded</div>
                </div>
              </div>

              {/* Direction Controls */}
              <div className="h-32 bg-muted border-t flex items-center justify-center">
                <div className="grid grid-cols-3 gap-4 w-48">
                  <div></div>
                  <Button variant="outline" size="default">FRONT</Button>
                  <div></div>
                  <Button variant="outline" size="default">LEFT</Button>
                  <div></div>
                  <Button variant="outline" size="default">RIGHT</Button>
                  <div></div>
                  <Button variant="outline" size="default">BACK</Button>
                  <div></div>
                </div>
              </div>
            </div>
          </div>
        </TabsContent>
        
        <TabsContent value="gates" className="flex-1 flex m-0 p-0 data-[state=active]:flex h-full">
          {/* Gates & Cameras Management */}
          <div className="flex h-full w-full min-h-0">
            <div className="flex-1 p-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h1 className="text-2xl font-bold">Gates & Cameras Management</h1>
                  <p className="text-muted-foreground">Manage your gates and associated cameras</p>
                </div>
                <Button onClick={handleAddGate}>
                  Add New Gate
                </Button>
              </div>
              
              <div className="grid gap-6">
                {gates.map((gate) => (
                  <Card key={gate.id} className="w-full">
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <CardTitle className="text-lg">{gate.name}</CardTitle>
                          <Badge variant={gate.status === 'active' ? 'default' : 'secondary'}>
                            {gate.status}
                          </Badge>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Button 
                            variant="outline" 
                            size="sm"
                            onClick={() => handleToggleGateStatus(gate.id)}
                          >
                            {gate.status === 'active' ? 'Deactivate' : 'Activate'}
                          </Button>
                          <Button 
                            variant="outline" 
                            size="sm"
                            onClick={() => handleEditGate(gate)}
                          >
                            Edit
                          </Button>
                          <Button 
                            variant="destructive" 
                            size="sm"
                            onClick={() => handleDeleteGate(gate.id)}
                          >
                            Delete
                          </Button>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="grid md:grid-cols-2 gap-6">
                        {/* Input Camera */}
                        <div className="space-y-3">
                          <div className="flex items-center space-x-2">
                            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                            <h3 className="font-medium">Input Camera</h3>
                          </div>
                          <div className="space-y-2">
                            <div>
                              <Label className="text-sm font-medium">Camera Name</Label>
                              <p className="text-sm text-muted-foreground">{gate.inputCamera.name}</p>
                            </div>
                            <div>
                              <Label className="text-sm font-medium">RTSP URL</Label>
                              <p className="text-sm text-muted-foreground font-mono">{gate.inputCamera.rtspUrl}</p>
                            </div>
                            <div>
                              <Label className="text-sm font-medium">Status</Label>
                              <Badge variant={gate.inputCamera.status === 'active' ? 'default' : 'secondary'} className="text-xs">
                                {gate.inputCamera.status}
                              </Badge>
                            </div>
                          </div>
                        </div>
                        
                        {/* Output Camera */}
                        <div className="space-y-3">
                          <div className="flex items-center space-x-2">
                            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                            <h3 className="font-medium">Output Camera</h3>
                          </div>
                          <div className="space-y-2">
                            <div>
                              <Label className="text-sm font-medium">Camera Name</Label>
                              <p className="text-sm text-muted-foreground">{gate.outputCamera.name}</p>
                            </div>
                            <div>
                              <Label className="text-sm font-medium">RTSP URL</Label>
                              <p className="text-sm text-muted-foreground font-mono">{gate.outputCamera.rtspUrl}</p>
                            </div>
                            <div>
                              <Label className="text-sm font-medium">Status</Label>
                              <Badge variant={gate.outputCamera.status === 'active' ? 'default' : 'secondary'} className="text-xs">
                                {gate.outputCamera.status}
                              </Badge>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {gate.description && (
                        <div className="mt-4 pt-4 border-t">
                          <Label className="text-sm font-medium">Description</Label>
                          <p className="text-sm text-muted-foreground mt-1">{gate.description}</p>
                        </div>
                      )}
                      
                      <div className="mt-4 pt-4 border-t text-xs text-muted-foreground">
                        Created: {gate.createdAt}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </div>
        </TabsContent>
      </Tabs>

      {/* Recognition Settings Dialog */}
      <Dialog open={showSettingsDialog} onOpenChange={setShowSettingsDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Recognition Settings</DialogTitle>
          </DialogHeader>
          
          <div className="space-y-6">
            <div>
              <Label className="text-sm">Tracker</Label>
              <Select value={tracker} onValueChange={setTracker}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="bytetrack">ByteTrack</SelectItem>
                  <SelectItem value="deepsort">DeepSort</SelectItem>
                  <SelectItem value="sort">SORT</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <Label className="text-sm">Confidence: {(confidence[0] / 100).toFixed(2)}</Label>
              <Slider
                value={confidence}
                onValueChange={setConfidence}
                max={100}
                min={0}
                step={1}
                className="mt-2"
              />
            </div>
            
            <div className="space-y-4">
              <Label className="text-sm font-medium">Display Options</Label>
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="detection-boxes"
                    checked={showDetectionBoxes}
                    onCheckedChange={setShowDetectionBoxes}
                  />
                  <Label htmlFor="detection-boxes" className="text-sm">Show Detection Boxes</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="track-trajectories"
                    checked={showTrackTrajectories}
                    onCheckedChange={setShowTrackTrajectories}
                  />
                  <Label htmlFor="track-trajectories" className="text-sm">Show Track Trajectories</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="recognition-names"
                    checked={showRecognitionNames}
                    onCheckedChange={setShowRecognitionNames}
                  />
                  <Label htmlFor="recognition-names" className="text-sm">Show Recognition Names</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="monitoring-regions"
                    checked={showMonitoringRegions}
                    onCheckedChange={setShowMonitoringRegions}
                  />
                  <Label htmlFor="monitoring-regions" className="text-sm">Show Monitoring Regions</Label>
                </div>
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Recognition Source Dialog */}
      <Dialog open={showSourceDialog} onOpenChange={setShowSourceDialog}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Recognition Source</DialogTitle>
          </DialogHeader>
          
          <div className="space-y-6">
            <RadioGroup value={recognitionSource} onValueChange={setRecognitionSource}>
              <div className="space-y-4">
                <div className="flex items-center space-x-2 p-3 rounded-lg border hover:bg-muted cursor-pointer">
                  <RadioGroupItem value="video" id="video" />
                  <Label htmlFor="video" className="cursor-pointer">Video File</Label>
                </div>
                
                {recognitionSource === 'video' && (
                  <div className="ml-6 space-y-2">
                    <Label htmlFor="video-path" className="text-sm">Video File Path:</Label>
                    <Input
                      id="video-path"
                      placeholder="Enter video file path (e.g., /path/to/video.mp4)"
                      value={videoFilePath}
                      onChange={(e) => setVideoFilePath(e.target.value)}
                      className="w-full"
                    />
                    <p className="text-xs text-muted-foreground">
                      Supported formats: MP4, AVI, MOV, MKV
                    </p>
                  </div>
                )}
                
                <div className="flex items-center space-x-2 p-3 rounded-lg border hover:bg-muted cursor-pointer">
                  <RadioGroupItem value="camera" id="camera" />
                  <Label htmlFor="camera" className="cursor-pointer">IP Camera</Label>
                </div>
                
                {recognitionSource === 'camera' && (
                  <div className="ml-6 space-y-2">
                    <Label htmlFor="rtsp-url" className="text-sm">RTSP URL:</Label>
                    <Input
                      id="rtsp-url"
                      placeholder="Enter RTSP URL (e.g., rtsp://192.168.1.100:554/stream)"
                      value={rtspUrl}
                      onChange={(e) => setRtspUrl(e.target.value)}
                      className="w-full"
                    />
                    <p className="text-xs text-muted-foreground">
                      Example: rtsp://username:password@ip:port/stream
                    </p>
                  </div>
                )}
              </div>
            </RadioGroup>
            
            <div className="text-sm text-muted-foreground text-center p-3 bg-muted rounded-lg">
              <div className="font-medium">Current Selection:</div>
              <div>{recognitionSource === 'video' ? 'Video File' : 'IP Camera'}</div>
              {recognitionSource === 'video' && videoFilePath && (
                <div className="text-xs mt-1 font-mono">{videoFilePath}</div>
              )}
              {recognitionSource === 'camera' && rtspUrl && (
                <div className="text-xs mt-1 font-mono">{rtspUrl}</div>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Gate Management Dialog */}
      <Dialog open={showGateDialog} onOpenChange={setShowGateDialog}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>
              {editingGate ? 'Edit Gate' : 'Add New Gate'}
            </DialogTitle>
          </DialogHeader>
          
          <div className="space-y-6">
            {/* Basic Information */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium">Basic Information</h3>
              <div className="grid grid-cols-1 gap-4">
                <div>
                  <Label htmlFor="gate-name">Gate Name</Label>
                  <Input
                    id="gate-name"
                    placeholder="Enter gate name"
                    value={gateForm.name}
                    onChange={(e) => setGateForm({...gateForm, name: e.target.value})}
                  />
                </div>
                <div>
                  <Label htmlFor="gate-description">Description</Label>
                  <Textarea
                    id="gate-description"
                    placeholder="Enter gate description"
                    value={gateForm.description}
                    onChange={(e) => setGateForm({...gateForm, description: e.target.value})}
                    rows={3}
                  />
                </div>
              </div>
            </div>
            
            {/* Input Camera */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium flex items-center">
                <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                Input Camera
              </h3>
              <div className="grid grid-cols-1 gap-4">
                <div>
                  <Label htmlFor="input-camera-name">Camera Name</Label>
                  <Input
                    id="input-camera-name"
                    placeholder="Enter input camera name"
                    value={gateForm.inputCameraName}
                    onChange={(e) => setGateForm({...gateForm, inputCameraName: e.target.value})}
                  />
                </div>
                <div>
                  <Label htmlFor="input-camera-rtsp">RTSP URL</Label>
                  <Input
                    id="input-camera-rtsp"
                    placeholder="rtsp://192.168.1.100:554/stream"
                    value={gateForm.inputCameraRtsp}
                    onChange={(e) => setGateForm({...gateForm, inputCameraRtsp: e.target.value})}
                  />
                </div>
              </div>
            </div>
            
            {/* Output Camera */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium flex items-center">
                <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
                Output Camera
              </h3>
              <div className="grid grid-cols-1 gap-4">
                <div>
                  <Label htmlFor="output-camera-name">Camera Name</Label>
                  <Input
                    id="output-camera-name"
                    placeholder="Enter output camera name"
                    value={gateForm.outputCameraName}
                    onChange={(e) => setGateForm({...gateForm, outputCameraName: e.target.value})}
                  />
                </div>
                <div>
                  <Label htmlFor="output-camera-rtsp">RTSP URL</Label>
                  <Input
                    id="output-camera-rtsp"
                    placeholder="rtsp://192.168.1.101:554/stream"
                    value={gateForm.outputCameraRtsp}
                    onChange={(e) => setGateForm({...gateForm, outputCameraRtsp: e.target.value})}
                  />
                </div>
              </div>
            </div>
            
            {/* Actions */}
            <div className="flex justify-end space-x-2 pt-4 border-t">
              <Button variant="outline" onClick={() => setShowGateDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleSaveGate}>
                {editingGate ? 'Update Gate' : 'Create Gate'}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}