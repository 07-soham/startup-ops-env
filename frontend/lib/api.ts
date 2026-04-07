import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface EmailInput {
  text: string;
  sender?: string;
  subject?: string;
  thread_id?: string;
}

export interface TaskInput {
  name: string;
  hours_required?: number;
  deadline?: number;
  priority?: string;
  effort?: number;
  impact?: number;
}

export interface SimulationRequest {
  mode: 'auto' | 'manual';
  scenario?: string;
  emails?: EmailInput[];
  tasks?: TaskInput[];
  seed?: number;
  max_steps?: number;
}

export interface ParsedEmail {
  text: string;
  urgency: string;
  sentiment: string;
  requires_action: boolean;
  thread_id: string;
  escalation_level: number;
}

export interface SimulationResponse {
  summary: string;
  emails: ParsedEmail[];
  actions: any[];
  result: string;
  confidence: string;
  total_reward: number;
  scores: {
    email_score: number;
    task_score: number;
    negotiation_score: number;
    overall_score: number;
  };
}

export interface Scenario {
  name: string;
  description: string;
  email_count: number;
  task_count: number;
  negotiation_count: number;
}

export const getScenarios = async (): Promise<Scenario[]> => {
  const response = await api.get('/scenarios');
  return response.data;
};

export const runSimulation = async (request: SimulationRequest): Promise<SimulationResponse> => {
  const response = await api.post('/run-simulation', request);
  return response.data;
};

export const parseEmail = async (email: EmailInput, context?: string[]) => {
  const response = await api.post('/parse-email', email, {
    params: { context },
  });
  return response.data;
};

export default api;
