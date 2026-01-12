import React, { createContext, useContext, useState, ReactNode } from "react";
import { hospitals as defaultHospitals, Hospital, MonthlyDistribution, historicalData } from "@/data/mockData";

interface DistributionContextType {
  hospitals: Hospital[];
  setHospitals: (hospitals: Hospital[]) => void;
  distributions: MonthlyDistribution[];
  addDistribution: (distribution: MonthlyDistribution) => void;
  selectedMonth: string;
  setSelectedMonth: (month: string) => void;
  selectedYear: number;
  setSelectedYear: (year: number) => void;
}

const DistributionContext = createContext<DistributionContextType | undefined>(undefined);

export const DistributionProvider = ({ children }: { children: ReactNode }) => {
  const [hospitals, setHospitals] = useState<Hospital[]>(defaultHospitals);
  const [distributions, setDistributions] = useState<MonthlyDistribution[]>(historicalData);
  const [selectedMonth, setSelectedMonth] = useState<string>("January");
  const [selectedYear, setSelectedYear] = useState<number>(2026);

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
        addDistribution,
        selectedMonth,
        setSelectedMonth,
        selectedYear,
        setSelectedYear,
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
