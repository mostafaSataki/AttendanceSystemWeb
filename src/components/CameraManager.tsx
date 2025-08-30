"use client";

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { 
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from '@/components/ui/dialog';
import { 
  Plus, 
  Camera, 
  Activity, 
  AlertCircle, 
  CheckCircle, 
  Eye,
  BarChart3,
  Loader2
} from 'lucide-react';
import HLSPlayer from './HLSPlayer';

interface Camera {
  camera_id: string;
  rtsp_url: string;
  name: string;
  is_active: boolean;
  analytics_enabled: boolean;
  last_frame_time?: string;
  analytics_consumers: number;
  has_current_frame: boolean;
}

interface CameraStats {
  frames_processed: number;
  faces_detected: number;
  faces_recognized: number;
  is_active: boolean;
  uptime_seconds: number;
  avg_processing_time: number;
}

export default function CameraManager() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);
  const [cameraStats, setCameraStats] = useState<Record<string, CameraStats>>({});
  
  // Add camera form state
  const [newCamera, setNewCamera] = useState({
    camera_id: '',
    name: '',
    rtsp_url: '',
    enable_analytics: true
  });

  const fetchCameraStatus = async () => {
    try {
      const response = await fetch('/api/stream/cameras/status');
      const data = await response.json();
      
      if (data.streams) {
        const cameraList = Object.entries(data.streams).map(([id, camera]: [string, any]) => ({
          camera_id: id,
          ...camera
        }));
        setCameras(cameraList);
      }
    } catch (err) {
      console.error('Error fetching camera status:', err);
      setError('Failed to fetch camera status');
    }
  };

  const fetchAnalyticsStats = async () => {
    try {
      const response = await fetch('/api/stream/analytics/stats');
      const data = await response.json();
      
      if (data.success && data.all_camera_stats) {
        setCameraStats(data.all_camera_stats);
      }
    } catch (err) {
      console.error('Error fetching analytics stats:', err);
    }
  };

  useEffect(() => {
    fetchCameraStatus();
    fetchAnalyticsStats();
    
    // Refresh every 5 seconds
    const interval = setInterval(() => {
      fetchCameraStatus();
      fetchAnalyticsStats();
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);

  const handleAddCamera = async () => {
    if (!newCamera.camera_id || !newCamera.name || !newCamera.rtsp_url) {
      setError('Please fill in all required fields');
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch('/api/stream/cameras/add', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newCamera)
      });

      const data = await response.json();
      
      if (data.success) {
        setIsAddDialogOpen(false);
        setNewCamera({
          camera_id: '',
          name: '',
          rtsp_url: '',
          enable_analytics: true
        });
        await fetchCameraStatus();
        setError(null);
      } else {
        setError(data.error || 'Failed to add camera');
      }
    } catch (err) {
      setError('Failed to add camera');
      console.error('Error adding camera:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleAnalytics = async (cameraId: string, enable: boolean) => {
    try {
      const endpoint = enable ? 'start' : 'stop';
      const response = await fetch(`/api/stream/cameras/${cameraId}/analytics/${endpoint}`, {
        method: 'POST'
      });
      
      const data = await response.json();
      if (!data.success) {
        setError(data.error || `Failed to ${endpoint} analytics`);
      } else {
        await fetchCameraStatus();
      }
    } catch (err) {
      setError(`Failed to toggle analytics`);
      console.error('Error toggling analytics:', err);
    }
  };

  const formatUptime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Camera Management</h2>
        
        <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
          <DialogTrigger asChild>
            <Button className="flex items-center gap-2">
              <Plus className="h-4 w-4" />
              Add Camera
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add New RTSP Camera</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <div>
                <Label htmlFor="camera-id">Camera ID *</Label>
                <Input
                  id="camera-id"
                  value={newCamera.camera_id}
                  onChange={(e) => setNewCamera({...newCamera, camera_id: e.target.value})}
                  placeholder="e.g., camera_001"
                />
              </div>
              <div>
                <Label htmlFor="camera-name">Camera Name *</Label>
                <Input
                  id="camera-name"
                  value={newCamera.name}
                  onChange={(e) => setNewCamera({...newCamera, name: e.target.value})}
                  placeholder="e.g., Main Entrance"
                />
              </div>
              <div>
                <Label htmlFor="rtsp-url">RTSP URL *</Label>
                <Input
                  id="rtsp-url"
                  value={newCamera.rtsp_url}
                  onChange={(e) => setNewCamera({...newCamera, rtsp_url: e.target.value})}
                  placeholder="rtsp://username:password@192.168.1.100/stream"
                />
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="enable-analytics"
                  checked={newCamera.enable_analytics}
                  onCheckedChange={(checked) => setNewCamera({...newCamera, enable_analytics: checked})}
                />
                <Label htmlFor="enable-analytics">Enable Analytics</Label>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsAddDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleAddCamera} disabled={isLoading}>
                {isLoading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                    Adding...
                  </>
                ) : (
                  'Add Camera'
                )}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {cameras.map((camera) => (
          <Card key={camera.camera_id}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Camera className="h-5 w-5" />
                  <CardTitle className="text-lg">{camera.name}</CardTitle>
                </div>
                <div className="flex items-center gap-2">
                  {camera.is_active ? (
                    <Badge variant="default" className="flex items-center gap-1">
                      <CheckCircle className="h-3 w-3" />
                      Active
                    </Badge>
                  ) : (
                    <Badge variant="secondary" className="flex items-center gap-1">
                      <AlertCircle className="h-3 w-3" />
                      Inactive
                    </Badge>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="text-sm text-muted-foreground">
                <p><strong>ID:</strong> {camera.camera_id}</p>
                <p><strong>URL:</strong> {camera.rtsp_url.substring(0, 50)}...</p>
                {camera.last_frame_time && (
                  <p><strong>Last Frame:</strong> {new Date(camera.last_frame_time).toLocaleTimeString()}</p>
                )}
              </div>

              {/* Analytics Stats */}
              {cameraStats[camera.camera_id] && (
                <div className="border rounded-lg p-3 space-y-2">
                  <div className="flex items-center gap-2 text-sm font-medium">
                    <BarChart3 className="h-4 w-4" />
                    Analytics Stats
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <p className="text-muted-foreground">Frames</p>
                      <p className="font-medium">{cameraStats[camera.camera_id].frames_processed}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Faces</p>
                      <p className="font-medium">{cameraStats[camera.camera_id].faces_detected}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Recognized</p>
                      <p className="font-medium">{cameraStats[camera.camera_id].faces_recognized}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Uptime</p>
                      <p className="font-medium">{formatUptime(cameraStats[camera.camera_id].uptime_seconds)}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Controls */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Analytics</span>
                  <Switch
                    checked={cameraStats[camera.camera_id]?.is_active || false}
                    onCheckedChange={(checked) => toggleAnalytics(camera.camera_id, checked)}
                    disabled={!camera.is_active}
                  />
                </div>
                
                <Button
                  className="w-full flex items-center gap-2"
                  onClick={() => setSelectedCamera(
                    selectedCamera === camera.camera_id ? null : camera.camera_id
                  )}
                  disabled={!camera.is_active}
                >
                  <Eye className="h-4 w-4" />
                  {selectedCamera === camera.camera_id ? 'Hide Stream' : 'View Stream'}
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Video Player */}
      {selectedCamera && (
        <div className="mt-8">
          <h3 className="text-xl font-semibold mb-4">Live Stream</h3>
          <HLSPlayer
            cameraId={selectedCamera}
            cameraName={cameras.find(c => c.camera_id === selectedCamera)?.name}
            onError={(error) => setError(error)}
            onSessionEnd={() => {/* Handle session end if needed */}}
            className="max-w-4xl mx-auto"
          />
        </div>
      )}

      {cameras.length === 0 && !isLoading && (
        <div className="text-center py-12">
          <Camera className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-lg font-semibold mb-2">No Cameras Added</h3>
          <p className="text-muted-foreground mb-4">
            Add your first RTSP camera to start streaming and analytics
          </p>
          <Button onClick={() => setIsAddDialogOpen(true)}>
            <Plus className="h-4 w-4 mr-2" />
            Add Camera
          </Button>
        </div>
      )}
    </div>
  );
}