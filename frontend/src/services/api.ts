/**
 * API Service para comunicação com o backend SNS AI
 */

const API_BASE_URL = "http://localhost:8000";

export interface PredictionRequest {
  month: number;
  year: number;
  stock_percentage: number;
}

export interface DistributionRequest {
  months: number[];
  year: number;
  stock_percentage: number;
  mode: string;
}

export interface HospitalData {
  institution: string;
  region: string;
  predicted_consumption: number;
  priority_weight: number;
  allocated_amount: number;
}

export interface DistributionResult {
  period: string;
  year: number;
  hospitals: HospitalData[];
  total_predicted: number;
  available_stock: number;
  optimization_score: number;
}

export interface PredictionResponse {
  month: number;
  year: number;
  total_predicted_consumption: number;
  hospital_count: number;
  hospital_predictions: Record<
    string,
    {
      predicted_consumption: number;
      region: string;
    }
  >;
  model_source: string;
}

export interface ModelMetrics {
  model_type: string;
  training_samples: number;
  test_samples: number;
  metrics: {
    r2_train: number;
    r2_test: number;
    mae_test: number;
    rmse_test: number;
    efficiency_percentage: number;
  };
  model_comparison: Record<
    string,
    {
      r2_train: number;
      r2_test: number;
    }
  >;
  timestamp: string;
}

export interface HospitalInfo {
  name: string;
  region: string;
  avg_monthly_consumption: number;
  data_points: number;
}

export interface HospitalsResponse {
  hospitals: HospitalInfo[];
  total_hospitals: number;
  regions: string[];
  data_source: string;
}

export interface ModelMetrics {
  model_type: string;
  training_samples: number;
  test_samples: number;
  metrics: {
    r2_train: number;
    r2_test: number;
    mae_test: number;
    rmse_test: number;
    efficiency_percentage: number;
  };
  model_comparison: Record<
    string,
    {
      r2_train: number;
      r2_test: number;
    }
  >;
  timestamp: string;
}

class APIService {
  private baseURL: string;

  constructor() {
    this.baseURL = API_BASE_URL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;

    const config: RequestInit = {
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return data as T;
    } catch (error) {
      console.error(`Request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  // Health check
  async healthCheck(): Promise<{ status: string; model_status: string }> {
    return this.request<{ status: string; model_status: string }>("/health");
  }

  // Predição de consumo
  async predictConsumption(
    request: PredictionRequest
  ): Promise<PredictionResponse> {
    return this.request<PredictionResponse>("/predict", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  // Otimização de distribuição
  async optimizeDistribution(
    request: DistributionRequest
  ): Promise<DistributionResult[]> {
    return this.request<DistributionResult[]>("/distribute", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  // Informações dos hospitais
  async getHospitals(): Promise<HospitalsResponse> {
    return this.request<HospitalsResponse>("/hospitals");
  }

  // Retreinar modelo
  async retrainModel(): Promise<{ message: string }> {
    return this.request<{ message: string }>("/retrain", {
      method: "POST",
    });
  }

  // Obter métricas do modelo
  async getModelMetrics(): Promise<ModelMetrics> {
    return this.request<ModelMetrics>("/model/metrics");
  }

  // Utilitário para converter mês nome para número
  static monthNameToNumber(monthName: string): number {
    const months = [
      "January",
      "February",
      "March",
      "April",
      "May",
      "June",
      "July",
      "August",
      "September",
      "October",
      "November",
      "December",
    ];
    return months.indexOf(monthName) + 1;
  }

  // Utilitário para converter número do mês para nome
  static monthNumberToName(monthNumber: number): string {
    const months = [
      "January",
      "February",
      "March",
      "April",
      "May",
      "June",
      "July",
      "August",
      "September",
      "October",
      "November",
      "December",
    ];
    return months[monthNumber - 1] || "Unknown";
  }

  // Validação de conexão com o backend
  async validateConnection(): Promise<boolean> {
    try {
      const health = await this.healthCheck();
      return health.status === "healthy";
    } catch {
      return false;
    }
  }

  // Obter métricas de eficácia do modelo
  async getModelMetrics(): Promise<ModelMetrics> {
    const response = await fetch(`${API_BASE_URL}/model/metrics`);
    if (!response.ok) {
      throw new Error(`Erro ao buscar métricas: ${response.statusText}`);
    }
    return response.json();
  }
}

// Instância singleton
export const apiService = new APIService();

// Hook customizado para React
export const useAPIService = () => {
  return apiService;
};
