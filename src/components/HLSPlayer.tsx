"use client";

import React, { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, Play, Square, AlertCircle } from 'lucide-react';

interface HLSPlayerProps {
  cameraId: string;
  cameraName?: string;
  onError?: (error: string) => void;
  onSessionEnd?: () => void;
  className?: string;
}

interface PlaybackSession {
  session_id: string;
  token: string;
  hls_url: string;
  expires_at: string;
}

export default function HLSPlayer({ 
  cameraId, 
  cameraName, 
  onError, 
  onSessionEnd,
  className = "" 
}: HLSPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const hlsRef = useRef<any>(null);
  
  const [isLoading, setIsLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [session, setSession] = useState<PlaybackSession | null>(null);
  const [hlsSupported, setHlsSupported] = useState(false);

  // Check HLS support
  useEffect(() => {
    const checkHlsSupport = async () => {
      if (typeof window !== 'undefined') {
        // Native HLS support (Safari)
        if (videoRef.current?.canPlayType('application/vnd.apple.mpegurl')) {
          setHlsSupported(true);
          return;
        }
        
        // Try to load hls.js for other browsers
        try {
          const HlsModule = await import('hls.js');
          const Hls = HlsModule.default;
          if (Hls.isSupported()) {
            setHlsSupported(true);
          } else {
            setError('HLS is not supported in this browser');
          }
        } catch (err) {
          setError('HLS.js library not available. Please install: npm install hls.js');
        }
      }
    };
    
    checkHlsSupport();
  }, []);

  const createPlaybackSession = async (): Promise<PlaybackSession | null> => {
    try {
      const response = await fetch('/api/stream/playback/create-session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          camera_id: cameraId,
          user_id: 'web-user' // In real app, get from auth
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to create session: ${response.status}`);
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Failed to create playback session');
      }

      return {
        session_id: data.session_id,
        token: data.token,
        hls_url: data.hls_url,
        expires_at: data.expires_at
      };
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Unknown error';
      console.error('Error creating playback session:', errorMsg);
      setError(errorMsg);
      onError?.(errorMsg);
      return null;
    }
  };

  const startHlsPlayback = async (sessionId: string) => {
    try {
      const response = await fetch(`/api/stream/playback/${sessionId}/start`, {
        method: 'POST'
      });

      if (!response.ok) {
        throw new Error(`Failed to start playback: ${response.status}`);
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Failed to start HLS playback');
      }

      return true;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to start HLS';
      console.error('Error starting HLS playback:', errorMsg);
      setError(errorMsg);
      return false;
    }
  };

  const loadHlsStream = async (hlsUrl: string, token: string) => {
    if (!videoRef.current || !hlsSupported) return;

    const video = videoRef.current;
    const fullHlsUrl = `http://localhost:8000${hlsUrl}`;

    try {
      // Native HLS support (Safari)
      if (video.canPlayType('application/vnd.apple.mpegurl')) {
        video.src = fullHlsUrl;
        video.load();
        await video.play();
      } else {
        // Use hls.js for other browsers
        const HlsModule = await import('hls.js');
        const Hls = HlsModule.default;
        
        if (hlsRef.current) {
          hlsRef.current.destroy();
        }

        const hls = new Hls({
          xhrSetup: (xhr: XMLHttpRequest) => {
            xhr.setRequestHeader('Authorization', `Bearer ${token}`);
          },
          maxBufferLength: 10, // Reduce buffer for lower latency
          maxMaxBufferLength: 20,
          liveSyncDurationCount: 3,
        });

        hlsRef.current = hls;
        
        hls.loadSource(fullHlsUrl);
        hls.attachMedia(video);
        
        hls.on(Hls.Events.MANIFEST_PARSED, () => {
          video.play().catch(err => {
            console.warn('Autoplay failed:', err);
          });
        });

        hls.on(Hls.Events.ERROR, (event, data) => {
          console.error('HLS Error:', data);
          if (data.fatal) {
            setError(`HLS Error: ${data.type} - ${data.details}`);
          }
        });
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to load stream';
      console.error('Error loading HLS stream:', errorMsg);
      setError(errorMsg);
    }
  };

  const startPlayback = async () => {
    if (!hlsSupported) {
      setError('HLS not supported in this browser');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Create playback session
      const newSession = await createPlaybackSession();
      if (!newSession) return;

      setSession(newSession);

      // Start HLS generation on server
      const hlsStarted = await startHlsPlayback(newSession.session_id);
      if (!hlsStarted) return;

      // Wait a moment for HLS segments to be generated
      await new Promise(resolve => setTimeout(resolve, 3000));

      // Load HLS stream in video player
      await loadHlsStream(newSession.hls_url, newSession.token);

      setIsPlaying(true);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to start playback';
      setError(errorMsg);
      onError?.(errorMsg);
    } finally {
      setIsLoading(false);
    }
  };

  const stopPlayback = async () => {
    setIsLoading(true);
    
    try {
      // Stop video
      if (videoRef.current) {
        videoRef.current.pause();
        videoRef.current.src = '';
      }

      // Cleanup hls.js
      if (hlsRef.current) {
        hlsRef.current.destroy();
        hlsRef.current = null;
      }

      // Stop server-side HLS generation
      if (session) {
        await fetch(`/api/stream/playback/${session.session_id}/stop`, {
          method: 'POST'
        });
      }

      setIsPlaying(false);
      setSession(null);
      onSessionEnd?.();
    } catch (err) {
      console.error('Error stopping playback:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (hlsRef.current) {
        hlsRef.current.destroy();
      }
      if (session) {
        fetch(`/api/stream/playback/${session.session_id}/stop`, { method: 'POST' })
          .catch(console.error);
      }
    };
  }, [session]);

  return (
    <Card className={`p-4 ${className}`}>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">
            {cameraName || `Camera ${cameraId}`}
          </h3>
          <div className="flex gap-2">
            {!isPlaying ? (
              <Button 
                onClick={startPlayback} 
                disabled={isLoading || !hlsSupported}
                className="flex items-center gap-2"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4" />
                    Play
                  </>
                )}
              </Button>
            ) : (
              <Button 
                onClick={stopPlayback} 
                disabled={isLoading}
                variant="destructive"
                className="flex items-center gap-2"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Stopping...
                  </>
                ) : (
                  <>
                    <Square className="h-4 w-4" />
                    Stop
                  </>
                )}
              </Button>
            )}
          </div>
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
          <video
            ref={videoRef}
            className="w-full h-full object-contain"
            controls={isPlaying}
            playsInline
            muted={false}
          >
            Your browser does not support video playback.
          </video>
          
          {!isPlaying && !error && (
            <div className="absolute inset-0 flex items-center justify-center text-white">
              <div className="text-center">
                <Play className="h-16 w-16 mx-auto mb-2 opacity-50" />
                <p className="text-sm opacity-75">Click Play to start streaming</p>
              </div>
            </div>
          )}
        </div>

        {session && (
          <div className="text-xs text-muted-foreground">
            <p>Session: {session.session_id}</p>
            <p>Expires: {new Date(session.expires_at).toLocaleString()}</p>
          </div>
        )}
      </div>
    </Card>
  );
}