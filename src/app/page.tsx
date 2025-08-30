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
  const [showEnrollmentSourceDialog, setShowEnrollmentSourceDialog] = useState(false)
  const [showPersonDialog, setShowPersonDialog] = useState(false)
  const [editingPerson, setEditingPerson] = useState(null)

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
  const [enrollmentVideoFilePath, setEnrollmentVideoFilePath] = useState('')
  const [enrollmentRtspUrl, setEnrollmentRtspUrl] = useState('')
  const [people, setPeople] = useState([
    {
      id: 1,
      first_name: "John",
      last_name: "Doe",
      personnel_code: "EMP001",
      department: "IT",
      position: "Software Engineer",
      email: "john.doe@company.com",
      phone: "+98-21-12345678",
      is_active: true,
      created_at: "2024-01-10T10:30:00Z",
      enrollments: [
        {
          id: 1,
          person_id: 1,
          face_encoding_path: "/uploads/encodings/emp001.pkl",
          face_image_path: "/uploads/images/emp001.jpg",
          confidence_score: 95,
          is_active: true,
          created_at: "2024-01-10T10:30:00Z"
        }
      ]
    },
    {
      id: 2,
      first_name: "Jane",
      last_name: "Smith",
      personnel_code: "EMP002",
      department: "HR",
      position: "HR Manager",
      email: "jane.smith@company.com",
      phone: "+98-21-87654321",
      is_active: true,
      created_at: "2024-01-11T14:20:00Z",
      enrollments: [
        {
          id: 2,
          person_id: 2,
          face_encoding_path: "/uploads/encodings/emp002.pkl",
          face_image_path: "/uploads/images/emp002.jpg",
          confidence_score: 87,
          is_active: true,
          created_at: "2024-01-11T14:20:00Z"
        }
      ]
    },
    {
      id: 3,
      first_name: "Ahmed",
      last_name: "Hassan",
      personnel_code: "EMP003",
      department: "Finance",
      position: "Financial Analyst",
      email: "ahmed.hassan@company.com",
      phone: "+98-21-11223344",
      is_active: false,
      created_at: "2024-01-12T09:15:00Z",
      enrollments: [
        {
          id: 3,
          person_id: 3,
          face_encoding_path: "/uploads/encodings/emp003.pkl",
          face_image_path: "/uploads/images/emp003.jpg",
          confidence_score: 92,
          is_active: false,
          created_at: "2024-01-12T09:15:00Z"
        }
      ]
    }
  ])
  const [personForm, setPersonForm] = useState({
    first_name: '',
    last_name: '',
    personnel_code: '',
    department: '',
    position: '',
    email: '',
    phone: ''
  })
  const [isEnrolling, setIsEnrolling] = useState(false)
      personnelCode: "EMP001",
      department: "IT Department",
      position: "Software Engineer",
      email: "john.doe@company.com",
      phone: "+98-21-1234-5678",
      enrollmentDate: "2024-01-10",
      lastSeen: "2024-01-15 14:30:25",
      status: "active",
      image: "/api/placeholder/64/64",
      confidence: 95
    },
    {
      id: 2,
      name: "Jane Smith",
      personnelCode: "EMP002",
      department: "HR Department",
      position: "HR Manager",
      email: "jane.smith@company.com",
      phone: "+98-21-1234-5679",
      enrollmentDate: "2024-01-12",
      lastSeen: "2024-01-15 14:28:12",
      status: "active",
      image: "/api/placeholder/64/64",
      confidence: 87
    },
    {
      id: 3,
      name: "Ahmed Hassan",
      personnelCode: "EMP003",
      department: "Finance Department",
      position: "Financial Analyst",
      email: "ahmed.hassan@company.com",
      phone: "+98-21-1234-5680",
      enrollmentDate: "2024-01-08",
      lastSeen: "2024-01-15 14:25:45",
      status: "inactive",
      image: "/api/placeholder/64/64",
      confidence: 92
    },
    {
      id: 4,
      name: "Maria Garcia",
      personnelCode: "EMP004",
      department: "Marketing Department",
      position: "Marketing Specialist",
      email: "maria.garcia@company.com",
      phone: "+98-21-1234-5681",
      enrollmentDate: "2024-01-05",
      lastSeen: "2024-01-15 14:22:33",
      status: "active",
      image: "/api/placeholder/64/64",
      confidence: 89
    }
  ])
  const [searchTerm, setSearchTerm] = useState('')
  const [filterStatus, setFilterStatus] = useState('all')
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
      image: "/api/placeholder/64/64",
      gateName: "Main Entrance",
      direction: "entry"
    },
    {
      id: 2,
      name: "Jane Smith", 
      time: "14:28:12",
      date: "2024-01-15",
      confidence: 87,
      image: "/api/placeholder/64/64",
      gateName: "Main Entrance",
      direction: "exit"
    },
    {
      id: 3,
      name: "Ahmed Hassan",
      time: "14:25:45", 
      date: "2024-01-15",
      confidence: 92,
      image: "/api/placeholder/64/64",
      gateName: "Parking Gate",
      direction: "entry"
    },
    {
      id: 4,
      name: "Maria Garcia",
      time: "14:22:33",
      date: "2024-01-15", 
      confidence: 89,
      image: "/api/placeholder/64/64",
      gateName: "Parking Gate",
      direction: "exit"
    },
    {
      id: 5,
      name: "Robert Johnson",
      time: "14:20:15",
      date: "2024-01-15", 
      confidence: 94,
      image: "/api/placeholder/64/64",
      gateName: "Main Entrance",
      direction: "entry"
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
      image: "/api/placeholder/64/64",
      gateName: "Main Entrance",
      direction: Math.random() > 0.5 ? "entry" : "exit"
    }
    setTransactions([newTransaction, ...transactions])
    setDetectionResults(['Person detected - ' + newTransaction.confidence + '% confidence'])
  }

  const handleClearDisplay = () => {
    setIsRecognizing(false)
    setDetectionResults([])
  }

  // Person CRUD Functions
  const getFilteredPeople = () => {
    return people.filter(person => {
      const matchesSearch = searchTerm === '' || 
        person.first_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        person.last_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        person.personnel_code.toLowerCase().includes(searchTerm.toLowerCase()) ||
        person.department.toLowerCase().includes(searchTerm.toLowerCase())
      
      const matchesStatus = filterStatus === 'all' || 
        (filterStatus === 'active' && person.is_active) ||
        (filterStatus === 'inactive' && !person.is_active)
      
      return matchesSearch && matchesStatus
    })
  }

  const handleAddPerson = () => {
    setEditingPerson(null)
    setPersonForm({
      first_name: '',
      last_name: '',
      personnel_code: '',
      department: '',
      position: '',
      email: '',
      phone: ''
    })
    setShowPersonDialog(true)
  }

  const handleEditPerson = (person) => {
    setEditingPerson(person)
    setPersonForm({
      first_name: person.first_name,
      last_name: person.last_name,
      personnel_code: person.personnel_code,
      department: person.department,
      position: person.position,
      email: person.email,
      phone: person.phone
    })
    setShowPersonDialog(true)
  }

  const handleSavePerson = () => {
    if (editingPerson) {
      // Update existing person
      const updatedPeople = people.map(person => 
        person.id === editingPerson.id 
          ? { ...person, ...personForm }
          : person
      )
      setPeople(updatedPeople)
    } else {
      // Add new person
      const newPerson = {
        id: Math.max(...people.map(p => p.id)) + 1,
        ...personForm,
        is_active: true,
        created_at: new Date().toISOString(),
        enrollments: []
      }
      setPeople([newPerson, ...people])
    }
    setShowPersonDialog(false)
  }

  const handleDeletePerson = (personId) => {
    if (confirm('Are you sure you want to delete this person? This will also delete all their enrollments.')) {
      setPeople(people.filter(person => person.id !== personId))
    }
  }

  const handleTogglePersonActive = (personId) => {
    setPeople(people.map(person => 
      person.id === personId 
        ? { ...person, is_active: !person.is_active }
        : person
    ))
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

  // Person CRUD Functions
  const handleAddPerson = () => {
    setEditingPerson(null)
    setPersonForm({
      name: '',
      personnelCode: '',
      department: '',
      position: '',
      email: '',
      phone: ''
    })
    setShowPersonDialog(true)
  }

  const handleEditPerson = (person) => {
    setEditingPerson(person)
    setPersonForm({
      name: person.name,
      personnelCode: person.personnelCode,
      department: person.department,
      position: person.position,
      email: person.email,
      phone: person.phone
    })
    setShowPersonDialog(true)
  }

  const handleSavePerson = () => {
    if (editingPerson) {
      // Update existing person
      const updatedPeople = people.map(person => 
        person.id === editingPerson.id 
          ? { 
              ...person, 
              first_name: personForm.first_name,
              last_name: personForm.last_name,
              personnel_code: personForm.personnel_code,
              department: personForm.department,
              position: personForm.position,
              email: personForm.email,
              phone: personForm.phone
            }
          : person
      )
      setPeople(updatedPeople)
    } else {
      // Add new person
      const newPerson = {
        id: Math.max(...people.map(p => p.id)) + 1,
        first_name: personForm.first_name,
        last_name: personForm.last_name,
        personnel_code: personForm.personnel_code,
        department: personForm.department,
        position: personForm.position,
        email: personForm.email,
        phone: personForm.phone,
        is_active: true,
        created_at: new Date().toISOString(),
        enrollments: []
      }
      setPeople([newPerson, ...people])
    }
    setShowPersonDialog(false)
  }

  const handleDeletePerson = (personId) => {
    if (confirm('Are you sure you want to delete this person? This will also delete all their enrollments.')) {
      setPeople(people.filter(person => person.id !== personId))
    }
  }

  const handleTogglePersonActive = (personId) => {
    setPeople(people.map(person => 
      person.id === personId 
        ? { ...person, is_active: !person.is_active }
        : person
    ))
  }

  // Filter and search functions
  const getFilteredPeople = () => {
    return people.filter(person => {
      const matchesSearch = searchTerm === '' || 
        person.first_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        person.last_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        person.personnel_code.toLowerCase().includes(searchTerm.toLowerCase()) ||
        person.department.toLowerCase().includes(searchTerm.toLowerCase())
      
      const matchesStatus = filterStatus === 'all' || 
        (filterStatus === 'active' && person.is_active) ||
        (filterStatus === 'inactive' && !person.is_active)
      
      return matchesSearch && matchesStatus
    })
  }

  return (
    <div className="flex h-screen bg-background">
      <Tabs value={mainTab} onValueChange={setMainTab} className="flex-1 flex flex-col h-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="recognition">Face Recognition</TabsTrigger>
          <TabsTrigger value="enrollment">Face Enrollment</TabsTrigger>
          <TabsTrigger value="gates">Gates & Cameras</TabsTrigger>
          <TabsTrigger value="traffic">Traffic Reports</TabsTrigger>
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
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-64">
                    <div className="space-y-3">
                      {transactions.slice(0, 5).map((transaction) => (
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
                              <div className="flex items-center justify-between text-xs text-muted-foreground mt-1">
                                <div className="flex items-center gap-2">
                                  <Badge variant={transaction.direction === 'entry' ? 'default' : 'secondary'} className="text-xs">
                                    {transaction.direction === 'entry' ? '📥 ورود' : '📤 خروج'}
                                  </Badge>
                                  <span>{transaction.gateName}</span>
                                </div>
                                <div className="flex flex-col items-end">
                                  <span>{transaction.date}</span>
                                  <span>{transaction.time}</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
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
            {/* Left Panel - Controls */}
            <div className="w-80 bg-card border-r p-4 flex flex-col gap-4 flex-shrink-0">
              {/* Enrollment Source Button */}
              <Button onClick={() => setShowEnrollmentSourceDialog(true)} className="w-full" variant="outline">
                Enrollment Source
              </Button>

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

            {/* Right Panel - People List */}
            <div className="flex-1 flex flex-col min-w-0">
              {/* Header */}
              <div className="p-4 border-b">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold">Enrolled People</h2>
                  <Button onClick={handleAddPerson}>
                    Add Person
                  </Button>
                </div>
                
                {/* Search and Filter */}
                <div className="flex gap-2">
                  <Input
                    placeholder="Search by name, code, department..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="flex-1"
                  />
                  <Select value={filterStatus} onValueChange={setFilterStatus}>
                    <SelectTrigger className="w-32">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All</SelectItem>
                      <SelectItem value="active">Active</SelectItem>
                      <SelectItem value="inactive">Inactive</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* People List */}
              <div className="flex-1 p-4">
                <ScrollArea className="h-full">
                  <div className="space-y-3">
                    {getFilteredPeople().map((person) => (
                      <Card key={person.id} className="p-4">
                        <div className="flex items-start space-x-4">
                          <Avatar className="w-16 h-16">
                            <AvatarImage 
                              src={person.enrollments && person.enrollments.length > 0 ? person.enrollments[0].face_image_path : ''} 
                              alt={`${person.first_name} ${person.last_name}`} 
                            />
                            <AvatarFallback className="text-lg">
                              {(person.first_name[0] + person.last_name[0]).toUpperCase()}
                            </AvatarFallback>
                          </Avatar>
                          
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between mb-2">
                              <div>
                                <h3 className="font-medium text-lg">{person.first_name} {person.last_name}</h3>
                                <p className="text-sm text-muted-foreground">{person.personnel_code}</p>
                              </div>
                              <div className="flex items-center gap-2">
                                <Badge variant={person.is_active ? 'default' : 'secondary'}>
                                  {person.is_active ? 'فعال' : 'غیرفعال'}
                                </Badge>
                                {person.enrollments && person.enrollments.length > 0 && (
                                  <Badge variant="outline" className="text-xs">
                                    {person.enrollments[0].confidence_score}%
                                  </Badge>
                                )}
                              </div>
                            </div>
                            
                            <div className="grid grid-cols-2 gap-2 text-sm text-muted-foreground mb-3">
                              <div>
                                <span className="font-medium">Department:</span> {person.department || 'N/A'}
                              </div>
                              <div>
                                <span className="font-medium">Position:</span> {person.position || 'N/A'}
                              </div>
                              <div>
                                <span className="font-medium">Email:</span> {person.email || 'N/A'}
                              </div>
                              <div>
                                <span className="font-medium">Phone:</span> {person.phone || 'N/A'}
                              </div>
                            </div>
                            
                            <div className="flex items-center justify-between text-xs text-muted-foreground">
                              <div>
                                <span className="font-medium">Enrolled:</span> {new Date(person.created_at).toLocaleDateString()}
                              </div>
                              <div>
                                <span className="font-medium">Enrollments:</span> {person.enrollments?.length || 0}
                              </div>
                            </div>
                            
                            <div className="flex gap-2 mt-3">
                              <Button 
                                size="sm" 
                                variant="outline"
                                onClick={() => handleEditPerson(person)}
                              >
                                Edit
                              </Button>
                              <Button 
                                size="sm" 
                                variant="outline"
                                onClick={() => handleTogglePersonActive(person.id)}
                              >
                                {person.is_active ? 'Deactivate' : 'Activate'}
                              </Button>
                              <Button 
                                size="sm" 
                                variant="destructive"
                                onClick={() => handleDeletePerson(person.id)}
                              >
                                Delete
                              </Button>
                            </div>
                          </div>
                        </div>
                      </Card>
                    ))}
                  </div>
                </ScrollArea>
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
        
        <TabsContent value="traffic" className="flex-1 flex m-0 p-0 data-[state=active]:flex h-full">
          {/* Traffic Reports */}
          <div className="flex h-full w-full min-h-0">
            {/* Left Sidebar - Filters */}
            <div className="w-80 bg-card border-r p-4 flex flex-col gap-4 flex-shrink-0">
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Traffic Filters</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label htmlFor="gate-filter">Gate</Label>
                    <Select value={trafficFilters.gate} onValueChange={(value) => setTrafficFilters({...trafficFilters, gate: value})}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select gate" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Gates</SelectItem>
                        {gates.map(gate => (
                          <SelectItem key={gate.id} value={gate.name}>{gate.name}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div>
                    <Label htmlFor="direction-filter">Direction</Label>
                    <Select value={trafficFilters.direction} onValueChange={(value) => setTrafficFilters({...trafficFilters, direction: value})}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select direction" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Directions</SelectItem>
                        <SelectItem value="entry">Entry</SelectItem>
                        <SelectItem value="exit">Exit</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div>
                    <Label htmlFor="status-filter">Status</Label>
                    <Select value={trafficFilters.status} onValueChange={(value) => setTrafficFilters({...trafficFilters, status: value})}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select status" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Status</SelectItem>
                        <SelectItem value="authorized">Authorized</SelectItem>
                        <SelectItem value="unauthorized">Unauthorized</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div>
                    <Label htmlFor="date-from">Date From</Label>
                    <Input
                      id="date-from"
                      type="date"
                      value={trafficFilters.dateFrom}
                      onChange={(e) => setTrafficFilters({...trafficFilters, dateFrom: e.target.value})}
                    />
                  </div>
                  
                  <div>
                    <Label htmlFor="date-to">Date To</Label>
                    <Input
                      id="date-to"
                      type="date"
                      value={trafficFilters.dateTo}
                      onChange={(e) => setTrafficFilters({...trafficFilters, dateTo: e.target.value})}
                    />
                  </div>
                  
                  <div>
                    <Label htmlFor="search">Search Person</Label>
                    <Input
                      id="search"
                      placeholder="Search by name..."
                      value={trafficFilters.search}
                      onChange={(e) => setTrafficFilters({...trafficFilters, search: e.target.value})}
                    />
                  </div>
                  
                  <div className="flex space-x-2">
                    <Button variant="outline" className="flex-1" onClick={() => setTrafficFilters({gate: 'all', direction: 'all', status: 'all', dateFrom: '', dateTo: '', search: ''})}>
                      Clear
                    </Button>
                    <Button className="flex-1">
                      Apply
                    </Button>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Export Options</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <Button className="w-full" variant="outline">
                    Export to CSV
                  </Button>
                  <Button className="w-full" variant="outline">
                    Export to PDF
                  </Button>
                  <Button className="w-full" variant="outline">
                    Print Report
                  </Button>
                </CardContent>
              </Card>
            </div>
            
            {/* Main Content */}
            <div className="flex-1 p-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h1 className="text-2xl font-bold">Traffic Reports</h1>
                  <p className="text-muted-foreground">Detailed attendance and access control reports</p>
                </div>
                <div className="flex space-x-2">
                  <Button variant="outline">
                    Refresh Data
                  </Button>
                  <Button>
                    Generate Report
                  </Button>
                </div>
              </div>
              
              {/* Summary Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-muted-foreground">Total Entries</p>
                        <p className="text-2xl font-bold">1,234</p>
                      </div>
                      <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                        <span className="text-green-600 text-sm">↗</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-muted-foreground">Total Exits</p>
                        <p className="text-2xl font-bold">1,198</p>
                      </div>
                      <div className="w-8 h-8 bg-red-100 rounded-full flex items-center justify-center">
                        <span className="text-red-600 text-sm">↘</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-muted-foreground">Authorized</p>
                        <p className="text-2xl font-bold">2,398</p>
                      </div>
                      <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                        <span className="text-blue-600 text-sm">✓</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-muted-foreground">Unauthorized</p>
                        <p className="text-2xl font-bold">34</p>
                      </div>
                      <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center">
                        <span className="text-orange-600 text-sm">!</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
              
              {/* Traffic Reports Table */}
              <Card>
                <CardHeader>
                  <CardTitle>Traffic Log</CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-96">
                    <div className="space-y-2">
                      {trafficReports.map((report) => (
                        <Card key={report.id} className="p-4">
                          <div className="flex items-center space-x-4">
                            <Avatar className="w-12 h-12">
                              <AvatarImage src={report.imageUrl} alt={report.personName} />
                              <AvatarFallback>
                                {report.personName.split(' ').map(n => n[0]).join('')}
                              </AvatarFallback>
                            </Avatar>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center justify-between">
                                <p className="text-sm font-medium">{report.personName}</p>
                                <Badge variant={report.status === 'authorized' ? 'default' : 'destructive'}>
                                  {report.status}
                                </Badge>
                              </div>
                              <div className="flex items-center justify-between text-xs text-muted-foreground mt-1">
                                <span>{report.gateName}</span>
                                <span className="flex items-center">
                                  {report.direction === 'entry' ? '↗' : '↘'} {report.direction}
                                </span>
                                <span>{report.timestamp}</span>
                              </div>
                              <div className="flex items-center justify-between text-xs text-muted-foreground mt-1">
                                <span>Camera: {report.cameraName}</span>
                                <span>Confidence: {report.confidence}%</span>
                              </div>
                            </div>
                          </div>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
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

      {/* Enrollment Source Dialog */}
      <Dialog open={showEnrollmentSourceDialog} onOpenChange={setShowEnrollmentSourceDialog}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Enrollment Source</DialogTitle>
          </DialogHeader>
          
          <div className="space-y-6">
            <RadioGroup value={enrollmentSource} onValueChange={setEnrollmentSource}>
              <div className="space-y-4">
                <div className="flex items-center space-x-2 p-3 rounded-lg border hover:bg-muted cursor-pointer">
                  <RadioGroupItem value="video" id="enrollment-video" />
                  <Label htmlFor="enrollment-video" className="cursor-pointer">Video File</Label>
                </div>
                
                {enrollmentSource === 'video' && (
                  <div className="ml-6 space-y-2">
                    <Label htmlFor="enrollment-video-path" className="text-sm">Video File Path:</Label>
                    <Input
                      id="enrollment-video-path"
                      placeholder="Enter video file path (e.g., /path/to/video.mp4)"
                      value={enrollmentVideoFilePath}
                      onChange={(e) => setEnrollmentVideoFilePath(e.target.value)}
                      className="w-full"
                    />
                    <p className="text-xs text-muted-foreground">
                      Supported formats: MP4, AVI, MOV, MKV
                    </p>
                  </div>
                )}
                
                <div className="flex items-center space-x-2 p-3 rounded-lg border hover:bg-muted cursor-pointer">
                  <RadioGroupItem value="camera" id="enrollment-camera" />
                  <Label htmlFor="enrollment-camera" className="cursor-pointer">IP Camera</Label>
                </div>
                
                {enrollmentSource === 'camera' && (
                  <div className="ml-6 space-y-2">
                    <Label htmlFor="enrollment-rtsp-url" className="text-sm">RTSP URL:</Label>
                    <Input
                      id="enrollment-rtsp-url"
                      placeholder="Enter RTSP URL (e.g., rtsp://192.168.1.100:554/stream)"
                      value={enrollmentRtspUrl}
                      onChange={(e) => setEnrollmentRtspUrl(e.target.value)}
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
              <div>{enrollmentSource === 'video' ? 'Video File' : 'IP Camera'}</div>
              {enrollmentSource === 'video' && enrollmentVideoFilePath && (
                <div className="text-xs mt-1 font-mono">{enrollmentVideoFilePath}</div>
              )}
              {enrollmentSource === 'camera' && enrollmentRtspUrl && (
                <div className="text-xs mt-1 font-mono">{enrollmentRtspUrl}</div>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Person Management Dialog */}
      <Dialog open={showPersonDialog} onOpenChange={setShowPersonDialog}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>
              {editingPerson ? 'Edit Person' : 'Add New Person'}
            </DialogTitle>
          </DialogHeader>
          
          <div className="space-y-6">
            {/* Personal Information */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium">Personal Information</h3>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="person-name">Full Name *</Label>
                  <Input
                    id="person-name"
                    placeholder="Enter full name"
                    value={personForm.name}
                    onChange={(e) => setPersonForm({...personForm, name: e.target.value})}
                    className="w-full"
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="personnel-code">Personnel Code *</Label>
                  <Input
                    id="personnel-code"
                    placeholder="Enter personnel code"
                    value={personForm.personnelCode}
                    onChange={(e) => setPersonForm({...personForm, personnelCode: e.target.value})}
                    className="w-full"
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="department">Department *</Label>
                  <Input
                    id="department"
                    placeholder="Enter department"
                    value={personForm.department}
                    onChange={(e) => setPersonForm({...personForm, department: e.target.value})}
                    className="w-full"
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="position">Position *</Label>
                  <Input
                    id="position"
                    placeholder="Enter position"
                    value={personForm.position}
                    onChange={(e) => setPersonForm({...personForm, position: e.target.value})}
                    className="w-full"
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="email">Email *</Label>
                  <Input
                    id="email"
                    type="email"
                    placeholder="Enter email address"
                    value={personForm.email}
                    onChange={(e) => setPersonForm({...personForm, email: e.target.value})}
                    className="w-full"
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="phone">Phone *</Label>
                  <Input
                    id="phone"
                    placeholder="Enter phone number"
                    value={personForm.phone}
                    onChange={(e) => setPersonForm({...personForm, phone: e.target.value})}
                    className="w-full"
                  />
                </div>
              </div>
            </div>
            
            {/* Current Status */}
            {editingPerson && (
              <div className="text-sm text-muted-foreground text-center p-3 bg-muted rounded-lg">
                <div className="font-medium">Current Status:</div>
                <div>{editingPerson.status === 'active' ? 'فعال' : 'غیرفعال'}</div>
                <div className="text-xs mt-1">Enrolled: {editingPerson.enrollmentDate}</div>
                {editingPerson.lastSeen && (
                  <div className="text-xs mt-1">Last Seen: {editingPerson.lastSeen}</div>
                )}
              </div>
            )}
          </div>
          
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => setShowPersonDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleSavePerson}>
              {editingPerson ? 'Update' : 'Add'} Person
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Person Management Dialog */}
      <Dialog open={showPersonDialog} onOpenChange={setShowPersonDialog}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>
              {editingPerson ? 'Edit Person' : 'Add New Person'}
            </DialogTitle>
          </DialogHeader>
          
          <div className="space-y-6">
            {/* Basic Information */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium">Basic Information</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="first_name">First Name *</Label>
                  <Input
                    id="first_name"
                    placeholder="Enter first name"
                    value={personForm.first_name}
                    onChange={(e) => setPersonForm({...personForm, first_name: e.target.value})}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="last_name">Last Name *</Label>
                  <Input
                    id="last_name"
                    placeholder="Enter last name"
                    value={personForm.last_name}
                    onChange={(e) => setPersonForm({...personForm, last_name: e.target.value})}
                  />
                </div>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="personnel_code">Personnel Code *</Label>
                <Input
                  id="personnel_code"
                  placeholder="Enter personnel code"
                  value={personForm.personnel_code}
                  onChange={(e) => setPersonForm({...personForm, personnel_code: e.target.value})}
                />
              </div>
            </div>

            {/* Job Information */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium">Job Information</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="department">Department</Label>
                  <Input
                    id="department"
                    placeholder="Enter department"
                    value={personForm.department}
                    onChange={(e) => setPersonForm({...personForm, department: e.target.value})}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="position">Position</Label>
                  <Input
                    id="position"
                    placeholder="Enter position"
                    value={personForm.position}
                    onChange={(e) => setPersonForm({...personForm, position: e.target.value})}
                  />
                </div>
              </div>
            </div>

            {/* Contact Information */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium">Contact Information</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input
                    id="email"
                    type="email"
                    placeholder="Enter email address"
                    value={personForm.email}
                    onChange={(e) => setPersonForm({...personForm, email: e.target.value})}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="phone">Phone</Label>
                  <Input
                    id="phone"
                    placeholder="Enter phone number"
                    value={personForm.phone}
                    onChange={(e) => setPersonForm({...personForm, phone: e.target.value})}
                  />
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex justify-end gap-2 pt-4">
              <Button variant="outline" onClick={() => setShowPersonDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleSavePerson}>
                {editingPerson ? 'Update Person' : 'Add Person'}
              </Button>
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