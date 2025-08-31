/**
 * API Service for Face Recognition Attendance System
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  // People Management APIs
  async getPeople(params?: { skip?: number; limit?: number; search?: string; status?: string }) {
    const queryParams = new URLSearchParams();
    if (params?.skip) queryParams.append('skip', params.skip.toString());
    if (params?.limit) queryParams.append('limit', params.limit.toString());
    if (params?.search) queryParams.append('search', params.search);
    if (params?.status) queryParams.append('status', params.status);

    const query = queryParams.toString();
    return this.request(`/api/people${query ? `?${query}` : ''}`);
  }

  async getPerson(personId: number) {
    return this.request(`/api/people/${personId}`);
  }

  async createPerson(personData: {
    first_name: string;
    last_name: string;
    personnel_code: string;
    department: string;
    position?: string;
    email?: string;
    phone?: string;
  }) {
    return this.request('/api/people', {
      method: 'POST',
      body: JSON.stringify(personData),
    });
  }

  async updatePerson(personId: number, personData: Partial<{
    first_name: string;
    last_name: string;
    personnel_code: string;
    department: string;
    position: string;
    email: string;
    phone: string;
    is_active: boolean;
  }>) {
    return this.request(`/api/people/${personId}`, {
      method: 'PUT',
      body: JSON.stringify(personData),
    });
  }

  async deletePerson(personId: number) {
    return this.request(`/api/people/${personId}`, {
      method: 'DELETE',
    });
  }

  async togglePersonStatus(personId: number) {
    return this.request(`/api/people/${personId}/toggle-status`, {
      method: 'POST',
    });
  }

  // Face Poses APIs
  async getPersonPoses(personId: number) {
    return this.request(`/api/poses/${personId}`);
  }

  async uploadPoseImage(personId: number, poseType: string, file: File) {
    const formData = new FormData();
    formData.append('file', file);
    
    return this.request(`/api/poses/${personId}/upload?pose_type=${poseType}`, {
      method: 'POST',
      headers: {}, // Remove content-type to let browser set it for FormData
      body: formData,
    });
  }

  async deletePose(personId: number, poseType: string) {
    return this.request(`/api/poses/${personId}/${poseType}`, {
      method: 'DELETE',
    });
  }

  async deleteAllPersonPoses(personId: number) {
    return this.request(`/api/poses/${personId}`, {
      method: 'DELETE',
    });
  }

  // Face Enrollment APIs
  async startEnrollment(source?: string, sourceConfig?: any, personName?: string) {
    // Use multi-pose endpoint since /start doesn't exist
    return this.startMultiPoseEnrollment(
      personName || `user_${Date.now()}`, 
      source || 'camera', 
      sourceConfig
    );
  }

  async startMultiPoseEnrollment(personName: string, source: string = 'camera', sourceConfig?: any) {
    return this.request('/api/enrollment/multi-pose', {
      method: 'POST',
      body: JSON.stringify({
        person_name: personName,
        source,
        source_config: sourceConfig || {},
      }),
    });
  }

  async stopEnrollment() {
    return this.request('/api/enrollment/stop', {
      method: 'POST',
    });
  }

  // Streaming Enrollment APIs
  async startEnrollmentStream(source: string = 'camera', sourceConfig?: any) {
    return this.request('/api/enrollment-stream/start-stream', {
      method: 'POST',
      body: JSON.stringify({
        source,
        source_config: sourceConfig || {},
      }),
    });
  }

  async stopEnrollmentStream(sessionId: string) {
    return this.request(`/api/enrollment-stream/stop-stream/${sessionId}`, {
      method: 'POST',
    });
  }

  async getStreamStatus(sessionId: string) {
    return this.request(`/api/enrollment-stream/status/${sessionId}`);
  }

  getStreamUrl(sessionId: string) {
    return `${this.baseUrl}/api/enrollment-stream/video/${sessionId}`;
  }

  async capturePose(poseType: string) {
    return this.request(`/api/enrollment/capture-pose?pose_type=${poseType}`, {
      method: 'POST',
    });
  }

  async confirmEnrollment(personId: number, posesData: any[]) {
    return this.request('/api/enrollment/confirm-enrollment', {
      method: 'POST',
      body: JSON.stringify({
        person_id: personId,
        poses_data: posesData,
      }),
    });
  }

  async uploadSingleImage(personId: number, file: File) {
    const formData = new FormData();
    formData.append('file', file);
    
    return this.request(`/api/enrollment/upload-single-image?person_id=${personId}`, {
      method: 'POST',
      headers: {}, // Remove content-type for FormData
      body: formData,
    });
  }

  // Face Recognition APIs
  async startRecognition(source: string = 'camera', sourceConfig?: any) {
    return this.request('/api/recognition/start', {
      method: 'POST',
      body: JSON.stringify({
        source,
        source_config: sourceConfig,
      }),
    });
  }

  async stopRecognition() {
    return this.request('/api/recognition/stop', {
      method: 'POST',
    });
  }

  async getCurrentDetections() {
    return this.request('/api/recognition/detections');
  }

  async getRecentTransactions(limit: number = 10) {
    return this.request(`/api/recognition/transactions?limit=${limit}`);
  }

  // Health Check
  async healthCheck() {
    return this.request('/health');
  }
}

// Create and export a singleton instance
export const apiService = new ApiService();
export default apiService;