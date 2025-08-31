'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Badge } from '@/components/ui/badge'
import { Checkbox } from '@/components/ui/checkbox'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Textarea } from '@/components/ui/textarea'
import { User, UserPlus, Camera, BarChart3, Users, Eye, ImagePlus, Images } from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import apiService from '@/services/api'
import { useEffect } from 'react'

export default function FaceRecognitionSystem() {
  const [mainTab, setMainTab] = useState('recognition')
  const [enrollmentTab, setEnrollmentTab] = useState('live-enrollment')
  const [collectedPoses, setCollectedPoses] = useState([])
  const [enrollmentPoses, setEnrollmentPoses] = useState([])
  const [personPoses, setPersonPoses] = useState([])
  const poseOrder = ['FRONT', 'LEFT', 'RIGHT', 'UP', 'DOWN']
  const [selectedPersonId, setSelectedPersonId] = useState('')
  const [reviewingPersonId, setReviewingPersonId] = useState(null)
  const [reviewMode, setReviewMode] = useState('enrollment') // 'enrollment' or 'person'
  const [hasCompletedEnrollment, setHasCompletedEnrollment] = useState(false)
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
  const [people, setPeople] = useState([])
  const [loading, setLoading] = useState(false)
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
  const [streamSessionId, setStreamSessionId] = useState('')
  const [streamUrl, setStreamUrl] = useState('')
  const [processingStartTime, setProcessingStartTime] = useState<number>(0)

  // Fetch people from API on component mount
  useEffect(() => {
    const fetchPeople = async () => {
      try {
        setLoading(true)
        const response = await apiService.getPeople()
        setPeople(Array.isArray(response) ? response : response.items || [])
      } catch (error) {
        console.error('Failed to fetch people:', error)
        setPeople([]) // Set empty array on error
      } finally {
        setLoading(false)
      }
    }
    
    fetchPeople()
  }, [])
  
  const employees = [
    {
      id: 1,
      name: "John Doe",
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
  ]
  
  const [searchTerm, setSearchTerm] = useState('')
  const [filterStatus, setFilterStatus] = useState('all')
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

  // Face samples functions
  const handleViewPersonPoses = async (personId) => {
    try {
      setLoading(true)
      console.log('Loading poses for person:', personId)
      
      // Fetch person's poses from backend
      const poses = await apiService.getPersonPoses(personId)
      console.log('Fetched poses:', poses)
      
      // Convert poses to numbered format and sort by creation time
      const numberedPoses = poses.map((pose, index) => ({
        id: pose.id || index + 1,
        number: index + 1,
        pose: pose.pose_type || `Sample ${index + 1}`,
        image: pose.image_path || '/api/placeholder/120/120',
        quality: pose.quality_score || 0,
        created_at: pose.created_at,
        type: 'saved' // Mark as saved pose
      }))
      
      setPersonPoses(numberedPoses)
      setReviewingPersonId(personId)
      setReviewMode('person')
      setEnrollmentTab('review-poses')
    } catch (error) {
      console.error('Failed to load person poses:', error)
      alert('Failed to load face samples. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleStopEnrollment = async () => {
    try {
      setLoading(true)
      setEnrollmentStatus('Stopping enrollment...')
      
      setIsEnrolling(false)
      setEnrollmentStatus('Enrollment stopped by user')
      
    } catch (error) {
      console.error('Failed to stop enrollment:', error)
      alert('Failed to stop enrollment. Please try again.')
    } finally {
      setIsEnrolling(false)
      setEnrollmentStatus('Ready')
      setLoading(false)
    }
  }

  const handleDeletePose = async (poseId, poseType) => {
    if (!confirm('Are you sure you want to delete this pose?')) {
      return
    }
    
    try {
      setLoading(true)
      
      if (reviewMode === 'person') {
        // Delete from backend for saved person poses
        await apiService.deletePose(reviewingPersonId, poseType)
        // Reload person poses
        await handleViewPersonPoses(reviewingPersonId)
      } else {
        // Remove from enrollment poses
        setEnrollmentPoses(poses => poses.filter(p => p.id !== poseId))
      }
    } catch (error) {
      console.error('Failed to delete pose:', error)
      alert('Failed to delete pose. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleAddSingleImage = (personId) => {
    // Create a file input element to select image
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'image/*'
    input.onchange = async (e) => {
      const file = e.target.files[0]
      if (file) {
        try {
          setLoading(true)
          console.log('Adding single image for person:', personId, 'File:', file.name)
          
          // Upload image to backend
          const result = await apiService.uploadSingleImage(personId, file)
          console.log('Upload result:', result)
          
          if (result.success) {
            alert('Image uploaded successfully!')
            // Refresh poses if we're currently viewing this person's poses
            if (reviewingPersonId === personId && reviewMode === 'person') {
              await handleViewPersonPoses(personId)
            }
          } else {
            alert('Failed to upload image. Please try again.')
          }
        } catch (error) {
          console.error('Failed to upload image:', error)
          alert('Failed to upload image. Please try again.')
        } finally {
          setLoading(false)
        }
      }
    }
    input.click()
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

  const handleSavePerson = async () => {
    try {
      setLoading(true)
      if (editingPerson) {
        // Update existing person
        await apiService.updatePerson(editingPerson.id, personForm)
        const response = await apiService.getPeople()
        setPeople(Array.isArray(response) ? response : response.items || [])
      } else {
        // Add new person
        await apiService.createPerson(personForm)
        const response = await apiService.getPeople()
        setPeople(Array.isArray(response) ? response : response.items || [])
      }
      setShowPersonDialog(false)
    } catch (error) {
      console.error('Failed to save person:', error)
      alert('Failed to save person. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleDeletePerson = async (personId) => {
    if (confirm('Are you sure you want to delete this person? This will also delete all their enrollments.')) {
      try {
        setLoading(true)
        await apiService.deletePerson(personId)
        const response = await apiService.getPeople()
        setPeople(Array.isArray(response) ? response : response.items || [])
      } catch (error) {
        console.error('Failed to delete person:', error)
        alert('Failed to delete person. Please try again.')
      } finally {
        setLoading(false)
      }
    }
  }

  const handleTogglePersonActive = async (personId) => {
    try {
      console.log('Toggling status for person:', personId)
      setLoading(true)
      
      const toggleResult = await apiService.togglePersonStatus(personId)
      console.log('Toggle result:', toggleResult)
      
      const response = await apiService.getPeople()
      console.log('Get people response:', response)
      
      const peopleArray = Array.isArray(response) ? response : response.items || []
      console.log('Setting people to:', peopleArray)
      setPeople(peopleArray)
      
    } catch (error) {
      console.error('Failed to toggle person status:', error)
      alert('Failed to update person status. Please try again.')
      // Don't clear the people list on error, just keep the current state
    } finally {
      setLoading(false)
    }
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


  const handleStartEnrollment = async () => {
    try {
      setLoading(true)
      setIsEnrolling(true)
      setEnrollmentStatus('Starting enrollment...')
      setHasCompletedEnrollment(false)
      setEnrollmentPoses([]) // Clear previous poses
      setProcessingStartTime(Date.now())
      
      // Prepare source configuration based on selected enrollment source
      let sourceConfig = {}
      let source = enrollmentSource
      
      if (enrollmentSource === 'video') {
        if (!enrollmentVideoFilePath.trim()) {
          alert('Please enter a video file path first')
          setIsEnrolling(false)
          setEnrollmentStatus('Ready')
          setLoading(false)
          return
        }
        sourceConfig = {
          video_path: enrollmentVideoFilePath.trim()
        }
        setEnrollmentStatus('Processing video file - this may take 1-2 minutes...')
        console.log('Starting enrollment with video file:', enrollmentVideoFilePath)
      } else if (enrollmentSource === 'camera') {
        if (!enrollmentRtspUrl.trim()) {
          alert('Please enter an RTSP URL first')
          setIsEnrolling(false)
          setEnrollmentStatus('Ready')
          setLoading(false)
          return
        }
        sourceConfig = {
          rtsp_url: enrollmentRtspUrl.trim()
        }
        setEnrollmentStatus('Connecting to RTSP stream - this may take 1-2 minutes...')
        console.log('Starting enrollment with RTSP camera:', enrollmentRtspUrl)
      } else {
        alert('Please select a video source and configure it')
        setIsEnrolling(false)
        setEnrollmentStatus('Ready')
        setLoading(false)
        return
      }
      
      // Call the multi-pose enrollment with timeout
      console.log('Calling backend with config:', { source, sourceConfig })
      console.log('API endpoint:', '/api/enrollment/multi-pose')
      console.log('Video file path:', sourceConfig.video_path)
      
      // Start progress timer
      const startTime = Date.now()
      const progressInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000)
        const minutes = Math.floor(elapsed / 60)
        const seconds = elapsed % 60
        setEnrollmentStatus(`Processing ${enrollmentSource === 'video' ? 'video file' : 'RTSP stream'}... (${minutes}:${seconds.toString().padStart(2, '0')})`)
      }, 1000)
      
      try {
        const enrollmentPromise = apiService.startMultiPoseEnrollment(`person_${Date.now()}`, source, sourceConfig)
        const timeoutPromise = new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Enrollment timeout - processing took too long (2+ minutes)')), 120000) // 2 minutes timeout
        )
        
        const result = await Promise.race([enrollmentPromise, timeoutPromise])
        console.log('Enrollment result:', result)
        
        clearInterval(progressInterval)
        
        // Process results
        setEnrollmentStatus('Processing completed')
        setIsEnrolling(false)
        
        if (result.pose_images && result.pose_images.length > 0) {
          const numberedPoses = result.pose_images.map((pose, index) => ({
            id: index + 1,
            number: index + 1,
            pose: pose.pose_name,
            image: pose.image,
            quality: pose.quality_score || 0,
            type: 'enrollment'
          }))
          setEnrollmentPoses(numberedPoses)
          setReviewMode('enrollment')
          setHasCompletedEnrollment(true)
          setEnrollmentTab('review-poses')
          setEnrollmentStatus(`Collected ${result.pose_images.length} poses - Review and confirm`)
        } else {
          setEnrollmentStatus('No poses collected - Check video source')
          alert('No poses were collected. Please check your video file path or RTSP URL.')
        }
      } catch (timeoutError) {
        clearInterval(progressInterval)
        console.error('Enrollment timeout or error:', timeoutError)
        setEnrollmentStatus('Processing failed - timeout or error')
        alert(`Enrollment failed: ${timeoutError.message}`)
      }
      
    } catch (error) {
      console.error('Failed to start enrollment:', error)
      alert(`Failed to start enrollment: ${error.message || 'Please try again.'}`)
      setIsEnrolling(false)
      setEnrollmentStatus('Ready')
    } finally {
      setLoading(false)
    }
  }

  const processEnrollmentWithBackend = async () => {
    // Prepare source configuration based on selected enrollment source
    let sourceConfig = {}
    let source = enrollmentSource
    
    if (enrollmentSource === 'video') {
      if (!enrollmentVideoFilePath.trim()) {
        alert('Please enter a video file path first')
        setIsEnrolling(false)
        setEnrollmentStatus('Ready')
        return
      }
      sourceConfig = {
        video_path: enrollmentVideoFilePath.trim()
      }
    } else if (enrollmentSource === 'camera') {
      if (!enrollmentRtspUrl.trim()) {
        // Use default camera (index 0) if no RTSP URL provided
        sourceConfig = {
          camera_index: 0
        }
      } else {
        sourceConfig = {
          rtsp_url: enrollmentRtspUrl.trim()
        }
      }
    } else {
      // Default to camera
      source = 'camera'
      sourceConfig = {
        camera_index: 0
      }
    }
    
    // Call the multi-pose enrollment
    const result = await apiService.startMultiPoseEnrollment(`person_${Date.now()}`, source, sourceConfig)
    console.log('Enrollment result:', result)
    
    // Stop webcam
    stopWebcamStream()
    
    // Process results
    setEnrollmentStatus('Processing completed')
    setIsEnrolling(false)
    
    if (result.pose_images && result.pose_images.length > 0) {
      const numberedPoses = result.pose_images.map((pose, index) => ({
        id: index + 1,
        number: index + 1,
        pose: pose.pose_name,
        image: pose.image,
        quality: pose.quality_score || 0,
        type: 'enrollment'
      }))
      setEnrollmentPoses(numberedPoses)
      setReviewMode('enrollment')
      setHasCompletedEnrollment(true)
      setEnrollmentTab('review-poses')
      setEnrollmentStatus(`Collected ${result.pose_images.length} poses - Review and confirm`)
    } else {
      setEnrollmentStatus('No poses collected - Try again')
      alert('No poses were collected. Please ensure your face is visible and try again.')
    }
  }

  const handleClearEnrollmentDisplay = async () => {
    setIsEnrolling(false)
    setEnrollmentStatus('Ready')
    setHasCompletedEnrollment(false)
    setEnrollmentPoses([])
    setPersonPoses([])
    setReviewMode('enrollment')
    setReviewingPersonId(null)
    setSelectedPersonId('')
    setStreamSessionId('')
    setStreamUrl('')
    setEnrollmentTab('live-enrollment') // Return to enrollment tab
  }

  // Person CRUD Functions

  // Filter and search functions

  return (
    <div className="flex h-screen bg-background">
      {/* Vertical Sidebar Menu */}
      <div className="w-64 bg-card border-r flex flex-col">
        <div className="p-6 border-b">
          <h2 className="text-lg font-semibold">Attendance System</h2>
        </div>
        <nav className="flex-1 p-4">
          <div className="space-y-2">
            <button
              onClick={() => setMainTab('recognition')}
              className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${
                mainTab === 'recognition'
                  ? 'bg-primary text-primary-foreground'
                  : 'hover:bg-accent hover:text-accent-foreground'
              }`}
            >
              <div className="flex items-center gap-3">
                <User className="w-5 h-5" />
                <span>Face Recognition</span>
              </div>
            </button>
            <button
              onClick={() => setMainTab('enrollment')}
              className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${
                mainTab === 'enrollment'
                  ? 'bg-primary text-primary-foreground'
                  : 'hover:bg-accent hover:text-accent-foreground'
              }`}
            >
              <div className="flex items-center gap-3">
                <UserPlus className="w-5 h-5" />
                <span>Face Enrollment</span>
              </div>
            </button>
            <button
              onClick={() => setMainTab('gates')}
              className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${
                mainTab === 'gates'
                  ? 'bg-primary text-primary-foreground'
                  : 'hover:bg-accent hover:text-accent-foreground'
              }`}
            >
              <div className="flex items-center gap-3">
                <Camera className="w-5 h-5" />
                <span>Gates & Cameras</span>
              </div>
            </button>
            <button
              onClick={() => setMainTab('traffic')}
              className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${
                mainTab === 'traffic'
                  ? 'bg-primary text-primary-foreground'
                  : 'hover:bg-accent hover:text-accent-foreground'
              }`}
            >
              <div className="flex items-center gap-3">
                <BarChart3 className="w-5 h-5" />
                <span>Traffic Reports</span>
              </div>
            </button>
          </div>
        </nav>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-full">
        {mainTab === 'recognition' && (
          <div className="flex-1 flex h-full">
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
          </div>
        )}
        
        {mainTab === 'enrollment' && (
          <div className="flex-1 flex flex-col h-full">
            {/* Enrollment Tabs */}
            <Tabs value={enrollmentTab} onValueChange={setEnrollmentTab} className="flex-1 flex flex-col h-full">
              <div className="border-b px-6 pt-4">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="live-enrollment" className="flex items-center gap-2">
                    <UserPlus className="w-4 h-4" />
                    <span>Live Enrollment</span>
                  </TabsTrigger>
                  <TabsTrigger value="review-poses" className="flex items-center gap-2">
                    <Eye className="w-4 h-4" />
                    <span>Review Poses</span>
                  </TabsTrigger>
                  <TabsTrigger value="people-management" className="flex items-center gap-2">
                    <Users className="w-4 h-4" />
                    <span>People Management</span>
                  </TabsTrigger>
                </TabsList>
              </div>

              <TabsContent value="live-enrollment" className="flex-1 flex m-0 p-0">
                {/* Live Enrollment Interface */}
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
                        {enrollmentSource === 'video' && enrollmentVideoFilePath && (
                          <Button 
                            onClick={async () => {
                              try {
                                const response = await fetch(`http://localhost:8001/api/video/info?video_path=${encodeURIComponent(enrollmentVideoFilePath)}`)
                                if (response.ok) {
                                  const info = await response.json()
                                  alert(`Video Info:\n• Duration: ${Math.round(info.duration_seconds)}s\n• Size: ${info.width}x${info.height}\n• FPS: ${Math.round(info.fps)}\n• File Size: ${info.size_mb}MB`)
                                } else {
                                  alert('Video file not found or cannot be opened!')
                                }
                              } catch (error) {
                                alert('Error checking video file!')
                              }
                            }}
                            variant="outline"
                            className="w-full mb-2"
                            size="sm"
                          >
                            Test Video File
                          </Button>
                        )}
                        
                        {!isEnrolling ? (
                          <Button 
                            onClick={handleStartEnrollment} 
                            className="w-full"
                            disabled={
                              (enrollmentSource === 'video' && !enrollmentVideoFilePath.trim()) ||
                              (enrollmentSource === 'camera' && !enrollmentRtspUrl.trim())
                            }
                          >
                            Start Enrollment
                          </Button>
                        ) : (
                          <Button 
                            onClick={handleStopEnrollment} 
                            variant="destructive"
                            className="w-full"
                          >
                            Stop Enrollment
                          </Button>
                        )}
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

                  {/* Right Panel - Video Preview */}
                  <div className="flex-1 bg-black flex items-center justify-center relative">
                    {isEnrolling ? (
                      <div className="w-full h-full flex flex-col items-center justify-center p-8">
                        {/* Video Info Display */}
                        {enrollmentSource === 'video' && enrollmentVideoFilePath && (
                          <div className="mb-4 bg-gray-800 rounded-lg p-4 max-w-md">
                            <div className="text-sm text-green-400 mb-2 text-center">📹 Video Processing Active</div>
                            <div className="text-xs text-gray-400 space-y-1">
                              <div><span className="text-white">File:</span> {enrollmentVideoFilePath.split('\\').pop()}</div>
                              <div><span className="text-white">Status:</span> Backend analyzing frames for face poses</div>
                              <div><span className="text-white">Progress:</span> Real-time face detection running</div>
                            </div>
                            <div className="mt-3 w-full bg-gray-700 rounded-full h-2">
                              <div className="bg-blue-500 h-2 rounded-full animate-pulse" style={{ width: '60%' }}></div>
                            </div>
                          </div>
                        )}
                        
                        {/* Processing Status */}
                        <div className="text-white text-center">
                          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
                          <div className="text-lg mb-2">{enrollmentStatus}</div>
                          <div className="text-sm text-gray-400 max-w-md">
                            {enrollmentSource === 'video' ? (
                              <>
                                <div className="font-mono text-xs break-all mb-2">
                                  {enrollmentVideoFilePath}
                                </div>
                                <div>Backend is analyzing video for face poses...</div>
                              </>
                            ) : (
                              <>
                                <div className="font-mono text-xs break-all mb-2">
                                  {enrollmentRtspUrl}
                                </div>
                                <div>Connecting to RTSP stream...</div>
                              </>
                            )}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="text-white text-center p-8 max-w-md">
                        <div className="text-lg mb-4">Video Source Enrollment</div>
                        
                        {/* Current Configuration Display */}
                        <div className="bg-gray-800 rounded-lg p-4 mb-4 text-left">
                          <div className="text-sm">
                            <div className="flex justify-between items-center mb-2">
                              <span className="text-gray-400">Source Type:</span>
                              <span className="capitalize">{enrollmentSource}</span>
                            </div>
                            
                            {enrollmentSource === 'video' ? (
                              <div className="flex justify-between items-center">
                                <span className="text-gray-400">Video Path:</span>
                                <span className="text-xs truncate ml-2 max-w-48">
                                  {enrollmentVideoFilePath || 'Not set'}
                                </span>
                              </div>
                            ) : (
                              <div className="flex justify-between items-center">
                                <span className="text-gray-400">RTSP URL:</span>
                                <span className="text-xs truncate ml-2 max-w-48">
                                  {enrollmentRtspUrl || 'Not set'}
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                        
                        {/* Video Info when configured */}
                        {enrollmentSource === 'video' && enrollmentVideoFilePath && (
                          <div className="mb-4 bg-gray-800 rounded-lg p-3">
                            <div className="text-sm text-gray-400 mb-2">📹 Video Ready:</div>
                            <div className="text-xs text-gray-300 space-y-1">
                              <div>File: {enrollmentVideoFilePath.split('\\').pop()}</div>
                              <div>Path: {enrollmentVideoFilePath}</div>
                              <div className="text-green-400">✓ Ready for enrollment</div>
                            </div>
                          </div>
                        )}
                        
                        <div className="text-sm text-gray-400">
                          {hasCompletedEnrollment 
                            ? "Enrollment completed - Review poses in the next tab" 
                            : "Configure source and click Start Enrollment"}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="review-poses" className="flex-1 flex m-0 p-0">
                {/* Review Poses Interface */}
                <div className="flex-1 p-6">
                  <div className="mb-6">
                    <h2 className="text-2xl font-bold">
                      {reviewMode === 'person' ? 'Face Samples' : 'Review Collected Poses'}
                    </h2>
                    <p className="text-muted-foreground">
                      {reviewMode === 'person' && reviewingPersonId ? 
                        `Viewing face samples for: ${people.find(p => p.id === reviewingPersonId)?.first_name} ${people.find(p => p.id === reviewingPersonId)?.last_name}` : 
                        reviewMode === 'enrollment' ? 
                        'Review and confirm enrollment poses' :
                        'Select a person to view their face samples'
                      }
                    </p>
                    {reviewingPersonId && (
                      <div className="flex gap-2 mt-2">
                        <Button 
                          variant="outline" 
                          size="sm" 
                          onClick={() => {
                            setReviewingPersonId(null)
                            setPersonPoses([])
                            setReviewMode('enrollment')
                          }}
                        >
                          Back to Enrollment
                        </Button>
                        {reviewMode === 'person' && (
                          <Button 
                            variant="outline" 
                            size="sm" 
                            onClick={() => handleAddSingleImage(reviewingPersonId)}
                            className="flex items-center gap-1"
                          >
                            <ImagePlus className="w-4 h-4" />
                            Add Sample
                          </Button>
                        )}
                      </div>
                    )}
                  </div>
                  
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-center">
                        {reviewMode === 'person' ? 'Face Samples' : 'Collected Poses'}
                      </CardTitle>
                      {reviewMode === 'person' && (
                        <p className="text-center text-sm text-muted-foreground">
                          Total: {personPoses.length} samples
                        </p>
                      )}
                    </CardHeader>
                    <CardContent className="p-6">
                      {/* Poses Grid - Numbered Layout */}
                      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6 mb-8 justify-items-center">
                        {(reviewMode === 'person' ? personPoses : enrollmentPoses).map((pose) => (
                          <div key={pose.id} className="flex flex-col items-center relative group">
                            {/* Pose Number Badge */}
                            <div className="absolute top-2 left-2 z-10 bg-primary text-primary-foreground text-xs font-bold rounded-full w-6 h-6 flex items-center justify-center">
                              {pose.number}
                            </div>
                            
                            {/* Delete Button */}
                            <Button
                              variant="destructive"
                              size="sm"
                              className="absolute top-2 right-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity w-6 h-6 p-0"
                              onClick={() => handleDeletePose(pose.id, pose.pose)}
                            >
                              ×
                            </Button>
                            
                            {/* Pose Image */}
                            <Card className={`w-32 h-40 overflow-hidden border-2 transition-colors ${
                              pose.type === 'enrollment' ? 'border-green-500 hover:border-green-600' : 
                              'border-blue-500 hover:border-blue-600'
                            }`}>
                              <div className="w-full h-full bg-muted flex items-center justify-center">
                                <img 
                                  src={pose.image || '/api/placeholder/128/160'} 
                                  alt={`${pose.pose} pose #${pose.number}`}
                                  className="w-full h-full object-cover"
                                />
                              </div>
                            </Card>
                            
                            {/* Pose Info */}
                            <div className="text-center mt-2">
                              <div className="font-medium text-sm">{pose.pose}</div>
                              {pose.quality !== undefined && (
                                <div className={`text-xs ${pose.type === 'enrollment' ? 'text-green-600' : 'text-blue-600'}`}>
                                  Quality: {pose.quality.toFixed(3)}
                                </div>
                              )}
                              {pose.created_at && (
                                <div className="text-xs text-muted-foreground">
                                  {new Date(pose.created_at).toLocaleDateString()}
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                        
                        {/* Empty State */}
                        {(reviewMode === 'person' ? personPoses : enrollmentPoses).length === 0 && (
                          <div className="col-span-full text-center py-8 text-muted-foreground">
                            {reviewMode === 'person' ? 
                              'No face samples found for this person' :
                              'No poses collected yet'
                            }
                          </div>
                        )}
                      </div>
                      
                      {/* Enrollment Confirmation - Show only in enrollment mode */}
                      {reviewMode === 'enrollment' && hasCompletedEnrollment && (
                        <Card className="bg-muted/50">
                          <CardHeader>
                            <CardTitle className="text-center text-lg">Confirm Enrollment</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="space-y-4">
                              <div>
                                <Label htmlFor="person-select">Enter Person's Name:</Label>
                                <Select value={selectedPersonId} onValueChange={setSelectedPersonId}>
                                  <SelectTrigger className="mt-1">
                                    <SelectValue placeholder="Select or search for a person..." />
                                  </SelectTrigger>
                                  <SelectContent>
                                    {getFilteredPeople().map((person) => (
                                      <SelectItem key={person.id} value={person.id.toString()}>
                                        {`${person.first_name} ${person.last_name}`} - {person.personnel_code}
                                      </SelectItem>
                                    ))}
                                    <SelectItem value="new-person">
                                      <div className="flex items-center gap-2">
                                        <UserPlus className="w-4 h-4" />
                                        <span>Add New Person</span>
                                      </div>
                                    </SelectItem>
                                  </SelectContent>
                                </Select>
                              </div>
                              <div className="flex gap-4 justify-center">
                                <Button 
                                  onClick={() => {
                                    if (selectedPersonId) {
                                      if (selectedPersonId === 'new-person') {
                                        // Handle new person enrollment
                                        console.log('Opening new person form...')
                                      } else {
                                        // Handle existing person enrollment
                                        const selectedPerson = people.find(p => p.id.toString() === selectedPersonId)
                                        console.log('Confirming enrollment for:', selectedPerson)
                                      }
                                      // Clear states after confirmation
                                      setSelectedPersonId('')
                                      setHasCompletedEnrollment(false)
                                      setEnrollmentStatus('Ready')
                                    }
                                  }}
                                  disabled={!selectedPersonId}
                                  className="bg-green-600 hover:bg-green-700"
                                >
                                  Confirm and Save
                                </Button>
                                <Button 
                                  onClick={() => {
                                    // Handle discard
                                    setSelectedPersonId('')
                                    setHasCompletedEnrollment(false)
                                    setEnrollmentStatus('Ready')
                                  }}
                                  variant="destructive"
                                >
                                  Discard
                                </Button>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      )}
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              <TabsContent value="people-management" className="flex-1 flex m-0 p-0">
                {/* People Management Interface (Current People List) */}
                <div className="flex-1 flex flex-col min-w-0">
                  {/* Header */}
                  <div className="p-4 border-b">
                    <div className="flex items-center justify-between mb-4">
                      <h2 className="text-lg font-semibold">People Management</h2>
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
                        {loading ? (
                          <div className="flex items-center justify-center py-8">
                            <div className="text-center">
                              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                              <p className="text-muted-foreground">Loading people...</p>
                            </div>
                          </div>
                        ) : people.length === 0 ? (
                          <div className="flex items-center justify-center py-8">
                            <div className="text-center">
                              <Users className="h-12 w-12 text-muted-foreground/50 mx-auto mb-4" />
                              <p className="text-muted-foreground">No people found</p>
                              <p className="text-sm text-muted-foreground">Add some people to get started</p>
                            </div>
                          </div>
                        ) : (
                          getFilteredPeople().map((person) => (
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
                            
                            <div className="flex flex-wrap gap-2 mt-3">
                              <Button 
                                size="sm" 
                                variant="outline"
                                onClick={() => handleViewPersonPoses(person.id)}
                                className="flex items-center gap-1"
                              >
                                <Images className="w-3 h-3" />
                                View Poses
                              </Button>
                              <Button 
                                size="sm" 
                                variant="outline"
                                onClick={() => handleAddSingleImage(person.id)}
                                className="flex items-center gap-1"
                              >
                                <ImagePlus className="w-3 h-3" />
                                Add Image
                              </Button>
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
                        ))
                        )}
                      </div>
                    </ScrollArea>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        )}
        
        {mainTab === 'gates' && (
          <div className="flex-1 flex h-full">
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
          </div>
        )}
        
        {mainTab === 'traffic' && (
          <div className="flex-1 flex h-full">
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
          </div>
        )}
      </div>

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
                      placeholder="D:/videos/face_enrollment.mp4"
                      value={enrollmentVideoFilePath}
                      onChange={(e) => setEnrollmentVideoFilePath(e.target.value)}
                      className="w-full font-mono text-sm"
                    />
                    <p className="text-xs text-muted-foreground">
                      Examples: D:/videos/sample.mp4, C:/Users/user/video.avi, /home/user/video.mp4
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Supported: MP4, AVI, MOV, MKV
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
                      placeholder="rtsp://192.168.1.100:554/stream"
                      value={enrollmentRtspUrl}
                      onChange={(e) => setEnrollmentRtspUrl(e.target.value)}
                      className="w-full font-mono text-sm"
                    />
                    <p className="text-xs text-muted-foreground">
                      Examples:<br/>
                      • rtsp://192.168.1.100:554/stream<br/>
                      • rtsp://admin:password@192.168.1.100:554/h264<br/>
                      • rtsp://camera.local/live/main
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
            
            {/* Dialog Actions */}
            <div className="flex justify-end gap-2 pt-4 border-t">
              <Button 
                variant="outline" 
                onClick={() => setShowEnrollmentSourceDialog(false)}
              >
                Cancel
              </Button>
              <Button 
                onClick={() => {
                  // Validate the form based on selected source
                  if (enrollmentSource === 'video' && !enrollmentVideoFilePath.trim()) {
                    alert('Please enter a video file path')
                    return
                  }
                  if (enrollmentSource === 'camera' && !enrollmentRtspUrl.trim()) {
                    alert('Please enter an RTSP URL')
                    return
                  }
                  
                  // Close dialog and apply settings
                  setShowEnrollmentSourceDialog(false)
                }}
              >
                OK
              </Button>
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