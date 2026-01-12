import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { useDistribution } from "@/context/DistributionContext";
import { hospitals } from "@/data/mockData";

const COLORS = [
  "hsl(199, 89%, 48%)",
  "hsl(172, 66%, 50%)",
  "hsl(221, 83%, 53%)",
  "hsl(262, 83%, 58%)",
  "hsl(330, 81%, 60%)",
  "hsl(25, 95%, 53%)",
];

const TimelineChart = () => {
  const { distributions } = useDistribution();
  const [selectedHospital, setSelectedHospital] = useState<string>("all");

  const chartData = useMemo(() => {
    return distributions.map(dist => {
      const dataPoint: Record<string, any> = {
        period: `${dist.month.slice(0, 3)} ${dist.year}`,
      };

      if (selectedHospital === "all") {
        hospitals.forEach(hospital => {
          const hospitalDist = dist.distributions.find(d => d.hospitalId === hospital.id);
          dataPoint[hospital.id] = hospitalDist?.quantity || 0;
        });
      } else {
        const hospitalDist = dist.distributions.find(d => d.hospitalId === selectedHospital);
        dataPoint[selectedHospital] = hospitalDist?.quantity || 0;
      }

      return dataPoint;
    });
  }, [distributions, selectedHospital]);

  const displayedHospitals = selectedHospital === "all" 
    ? hospitals 
    : hospitals.filter(h => h.id === selectedHospital);

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-lg font-semibold">Distribution Over Time</CardTitle>
        <Select value={selectedHospital} onValueChange={setSelectedHospital}>
          <SelectTrigger className="w-48">
            <SelectValue placeholder="Select hospital" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Hospitals</SelectItem>
            {hospitals.map(hospital => (
              <SelectItem key={hospital.id} value={hospital.id}>
                {hospital.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </CardHeader>
      <CardContent>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis 
                dataKey="period" 
                tick={{ fontSize: 12, fill: "hsl(var(--muted-foreground))" }}
              />
              <YAxis 
                tick={{ fontSize: 12, fill: "hsl(var(--muted-foreground))" }}
                tickFormatter={(value) => value.toLocaleString()}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                  boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                }}
                formatter={(value: number) => [value.toLocaleString() + " units"]}
              />
              <Legend />
              {displayedHospitals.map((hospital, index) => (
                <Line
                  key={hospital.id}
                  type="monotone"
                  dataKey={hospital.id}
                  name={hospital.name}
                  stroke={COLORS[hospitals.findIndex(h => h.id === hospital.id) % COLORS.length]}
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  activeDot={{ r: 6 }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export default TimelineChart;
