import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { useDistribution } from "@/context/DistributionContext";

const COLORS = [
  "hsl(199, 89%, 48%)",
  "hsl(172, 66%, 50%)",
  "hsl(221, 83%, 53%)",
  "hsl(262, 83%, 58%)",
  "hsl(330, 81%, 60%)",
  "hsl(25, 95%, 53%)",
];

const DistributionChart = () => {
  const { distributions, hospitals } = useDistribution();
  const [selectedPeriod, setSelectedPeriod] = useState<string>("all");

  const chartData = useMemo(() => {
    const filteredDistributions = selectedPeriod === "all"
      ? distributions
      : distributions.filter(d => `${d.month}-${d.year}` === selectedPeriod);

    const hospitalTotals: Record<string, number> = {};

    filteredDistributions.forEach(dist => {
      dist.distributions.forEach(d => {
        hospitalTotals[d.hospitalId] = (hospitalTotals[d.hospitalId] || 0) + d.quantity;
      });
    });

    return hospitals.map((hospital, index) => ({
      name: hospital.name.split(" ").slice(0, 2).join(" "),
      fullName: hospital.name,
      quantity: hospitalTotals[hospital.id] || 0,
      color: COLORS[index % COLORS.length],
    }));
  }, [distributions, selectedPeriod]);

  const periodOptions = [
    { value: "all", label: "All Time" },
    ...distributions.map(d => ({
      value: `${d.month}-${d.year}`,
      label: `${d.month} ${d.year}`,
    })),
  ];

  return (
    <Card className="col-span-2">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-lg font-semibold">Distribution by Hospital</CardTitle>
        <Select value={selectedPeriod} onValueChange={setSelectedPeriod}>
          <SelectTrigger className="w-40">
            <SelectValue placeholder="Select period" />
          </SelectTrigger>
          <SelectContent>
            {periodOptions.map(option => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </CardHeader>
      <CardContent>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 12, fill: "hsl(var(--muted-foreground))" }}
                angle={-45}
                textAnchor="end"
                height={80}
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
                formatter={(value: number, name: string, props: any) => [
                  value.toLocaleString() + " units",
                  props.payload.fullName,
                ]}
              />
              <Bar dataKey="quantity" radius={[6, 6, 0, 0]}>
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export default DistributionChart;
