"use client";

import CameraManager from '@/components/CameraManager';

export default function StreamingPage() {
  return (
    <div className="container mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">RTSP Camera Streaming</h1>
        <p className="text-muted-foreground mt-2">
          Manage RTSP cameras with real-time streaming and face recognition analytics
        </p>
      </div>
      
      <CameraManager />
    </div>
  );
}