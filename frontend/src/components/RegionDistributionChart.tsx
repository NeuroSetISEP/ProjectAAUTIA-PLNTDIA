import { useMemo, useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, PieChart, Pie } from "recharts";
import { apiService } from "@/services/api";

const COLORS = [
  "hsl(199, 89%, 48%)",
  "hsl(172, 66%, 50%)",
  "hsl(221, 83%, 53%)",
  "hsl(262, 83%, 58%)",
  "hsl(330, 81%, 60%)",
  "hsl(25, 95%, 53%)",
  "hsl(142, 71%, 45%)",
  "hsl(346, 77%, 49%)",
];

interface RegionData {
  region: string;
  predicted: number;
  allocated: number;
  hospitalCount: number;
  percentage: number;
}

const RegionDistributionChart = () => {
  const [chartType, setChartType] = useState<"bar" | "pie">("bar");
  const [regionData, setRegionData] = useState<RegionData[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchDistributionData = async () => {
    try {
      setLoading(true);

      // Fetch real distribution data from API
      const response = await apiService.optimizeDistribution({
        months: [new Date().getMonth() + 1],
        year: new Date().getFullYear(),
        stock_percentage: 0.7,
        mode: "month"
      });

      if (response && response.length > 0) {
        const distributionData = response[0];

        // Group by region
        const regionMap = new Map<string, RegionData>();

        distributionData.hospitals.forEach(hospital => {
          const region = hospital.region || 'Região Desconhecida';

          if (!regionMap.has(region)) {
            regionMap.set(region, {
              region,
              predicted: 0,
              allocated: 0,
              hospitalCount: 0,
              percentage: 0
            });
          }

          const regionInfo = regionMap.get(region)!;
          regionInfo.predicted += hospital.predicted_consumption;
          regionInfo.allocated += hospital.allocated_amount;
          regionInfo.hospitalCount += 1;
        });

        // Calculate percentages and convert to array
        const totalAllocated = Array.from(regionMap.values())
          .reduce((sum, region) => sum + region.allocated, 0);

        const regions = Array.from(regionMap.values()).map(region => ({
          ...region,
          percentage: Math.round((region.allocated / totalAllocated) * 100)
        })).sort((a, b) => b.allocated - a.allocated);

        setRegionData(regions);
      }
    } catch (error) {
      console.error("Erro ao buscar dados de distribuição:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDistributionData();
  }, []);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-card border rounded-lg p-3 shadow-lg">
          <p className="font-medium">{label}</p>
          <p className="text-blue-600">
            Hospitais: <span className="font-medium">{data.hospitalCount}</span>
          </p>
          <p className="text-green-600">
            Previsto: <span className="font-medium">{data.predicted.toFixed(1)}</span>
          </p>
          <p className="text-purple-600">
            Alocado: <span className="font-medium">{data.allocated}</span>
          </p>
          <p className="text-orange-600">
            Percentual: <span className="font-medium">{data.percentage}%</span>
          </p>
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <Card className="col-span-2">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <CardTitle className="text-lg font-semibold">Distribuição por Região</CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="h-80 flex items-center justify-center">
            <p className="text-muted-foreground">Carregando dados...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="col-span-2">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-lg font-semibold">Distribuição por Região de Saúde</CardTitle>
        <Select value={chartType} onValueChange={(value: "bar" | "pie") => setChartType(value)}>
          <SelectTrigger className="w-40">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="bar">Gráfico de Barras</SelectItem>
            <SelectItem value="pie">Gráfico Circular</SelectItem>
          </SelectContent>
        </Select>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            {chartType === "bar" ? (
              <BarChart data={regionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="region"
                  angle={-45}
                  textAnchor="end"
                  height={80}
                  fontSize={12}
                  stroke="hsl(var(--muted-foreground))"
                />
                <YAxis stroke="hsl(var(--muted-foreground))" />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="allocated" radius={[4, 4, 0, 0]}>
                  {regionData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            ) : (
              <PieChart>
                <Pie
                  data={regionData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ region, percentage }) => `${region.split(' ')[2] || region}: ${percentage}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="allocated"
                >
                  {regionData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            )}
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export default RegionDistributionChart;