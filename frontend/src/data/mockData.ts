export interface Hospital {
  id: string;
  name: string;
  allocationPercentage: number;
}

export interface MonthlyDistribution {
  month: string;
  year: number;
  totalDrugs: number;
  distributions: {
    hospitalId: string;
    quantity: number;
  }[];
}

export const hospitals: Hospital[] = [
  { id: "h1", name: "Central Medical Center", allocationPercentage: 25 },
  { id: "h2", name: "St. Mary's Hospital", allocationPercentage: 20 },
  { id: "h3", name: "City General Hospital", allocationPercentage: 18 },
  { id: "h4", name: "Regional Health Center", allocationPercentage: 15 },
  { id: "h5", name: "University Medical", allocationPercentage: 12 },
  { id: "h6", name: "Community Care Hospital", allocationPercentage: 10 },
];

export const months = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December"
];

export const generateDistributionData = (
  totalDrugs: number,
  hospitalsList: Hospital[]
): { hospitalId: string; hospitalName: string; quantity: number; percentage: number }[] => {
  return hospitalsList.map((hospital) => ({
    hospitalId: hospital.id,
    hospitalName: hospital.name,
    quantity: Math.round((totalDrugs * hospital.allocationPercentage) / 100),
    percentage: hospital.allocationPercentage,
  }));
};

// Historical mock data for visualization
export const historicalData: MonthlyDistribution[] = [
  {
    month: "October",
    year: 2025,
    totalDrugs: 50000,
    distributions: [
      { hospitalId: "h1", quantity: 12500 },
      { hospitalId: "h2", quantity: 10000 },
      { hospitalId: "h3", quantity: 9000 },
      { hospitalId: "h4", quantity: 7500 },
      { hospitalId: "h5", quantity: 6000 },
      { hospitalId: "h6", quantity: 5000 },
    ],
  },
  {
    month: "November",
    year: 2025,
    totalDrugs: 55000,
    distributions: [
      { hospitalId: "h1", quantity: 13750 },
      { hospitalId: "h2", quantity: 11000 },
      { hospitalId: "h3", quantity: 9900 },
      { hospitalId: "h4", quantity: 8250 },
      { hospitalId: "h5", quantity: 6600 },
      { hospitalId: "h6", quantity: 5500 },
    ],
  },
  {
    month: "December",
    year: 2025,
    totalDrugs: 60000,
    distributions: [
      { hospitalId: "h1", quantity: 15000 },
      { hospitalId: "h2", quantity: 12000 },
      { hospitalId: "h3", quantity: 10800 },
      { hospitalId: "h4", quantity: 9000 },
      { hospitalId: "h5", quantity: 7200 },
      { hospitalId: "h6", quantity: 6000 },
    ],
  },
];
