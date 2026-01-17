import React, { createContext, useContext, useState, ReactNode, useEffect } from "react";
import { hospitals as defaultHospitals, Hospital, MonthlyDistribution, historicalData } from "@/data/mockData";
import { apiService, type HospitalInfo } from "@/services/api";
import { useToast } from "@/hooks/use-toast";

interface DistributionContextType {
  hospitals: Hospital[];
  setHospitals: (hospitals: Hospital[]) => void;
  distributions: MonthlyDistribution[];
  setDistributions: (distributions: MonthlyDistribution[]) => void;
  selectedMonth: string;
  setSelectedMonth: (month: string) => void;
  selectedYear: number;
  setSelectedYear: (year: number) => void;
  addDistribution: (distribution: MonthlyDistribution) => void;
  apiHospitals: HospitalInfo[];
  isBackendConnected: boolean;
  loadAPIHospitals: () => Promise<void>;
}

const DistributionContext = createContext<DistributionContextType | undefined>(undefined);

export const DistributionProvider = ({ children }: { children: ReactNode }) => {
  const [hospitals, setHospitals] = useState<Hospital[]>([]);
  const [distributions, setDistributions] = useState<MonthlyDistribution[]>(historicalData);
  const [selectedMonth, setSelectedMonth] = useState<string>("January");
  const [selectedYear, setSelectedYear] = useState<number>(2026);
  const [apiHospitals, setAPIHospitals] = useState<HospitalInfo[]>([]);
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(false);

  const { toast } = useToast();

  // Verificar conexão com backend na inicialização
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const isConnected = await apiService.validateConnection();
        setIsBackendConnected(isConnected);

        if (isConnected) {
          console.log("✅ Backend conectado - carregando dados reais");
          // Carregar dados reais dos hospitais
          await loadAPIHospitals();
          toast({
            title: "Backend Connected",
            description: "✅ Successfully connected to SNS AI backend",
          });
        } else {
          console.log("⚠️ Backend offline, usando dados mock");
          setHospitals(defaultHospitals);
          toast({
            title: "Backend Offline",
            description: "⚠️ Using mock data. Start backend for real predictions.",
            variant: "destructive",
          });
        }
      } catch (error) {
        console.error("Backend connection check failed:", error);
        setIsBackendConnected(false);
      }
    };

    checkConnection();
  }, [toast]);

  const loadAPIHospitals = async () => {
    try {
      console.log("🔄 Carregando hospitais da API...");
      const response = await apiService.getHospitals();
      console.log("📊 Dados da API recebidos:", response.hospitals.length, "hospitais");
      setAPIHospitals(response.hospitals);

      // Converter hospitais da API para o formato local
      const totalConsumption = response.hospitals.reduce((sum, h) => sum + h.avg_monthly_consumption, 0);
      const convertedHospitals: Hospital[] = response.hospitals.map((h, index) => {
        const percentage = totalConsumption > 0
          ? Math.round((h.avg_monthly_consumption / totalConsumption) * 100)
          : Math.round(100 / response.hospitals.length); // Distribuição igual se não há dados

        return {
          id: `api_${index}`,
          name: h.name,
          allocationPercentage: Math.max(percentage, 1) // Mínimo 1%
        };
      });

      console.log("✅ Hospitais convertidos:", convertedHospitals.length);
      console.log("📋 Primeiros 3 hospitais:", convertedHospitals.slice(0, 3));

      // Forçar atualização do estado
      setHospitals([...convertedHospitals]);

      // Forçar re-render dos componentes
      setTimeout(() => {
        setHospitals([...convertedHospitals]);
      }, 100);

      console.log("🎉 Estado atualizado com", convertedHospitals.length, "hospitais");

      toast({
        title: "Success",
        description: `✅ Loaded ${convertedHospitals.length} hospitals from API`,
      });
    } catch (error) {
      console.error("Failed to load API hospitals:", error);
      toast({
        title: "Error",
        description: "Failed to load hospital data from backend",
        variant: "destructive",
      });
    }
  };

  const addDistribution = (distribution: MonthlyDistribution) => {
    setDistributions((prev) => {
      const existingIndex = prev.findIndex(
        (d) => d.month === distribution.month && d.year === distribution.year
      );
      if (existingIndex >= 0) {
        const updated = [...prev];
        updated[existingIndex] = distribution;
        return updated;
      }
      return [...prev, distribution];
    });
  };

  return (
    <DistributionContext.Provider
      value={{
        hospitals,
        setHospitals,
        distributions,
        setDistributions,
        addDistribution,
        selectedMonth,
        setSelectedMonth,
        selectedYear,
        setSelectedYear,
        apiHospitals,
        isBackendConnected,
        loadAPIHospitals,
      }}
    >
      {children}
    </DistributionContext.Provider>
  );
};

export const useDistribution = () => {
  const context = useContext(DistributionContext);
  if (!context) {
    throw new Error("useDistribution must be used within a DistributionProvider");
  }
  return context;
};
