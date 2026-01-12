import { ReactNode } from "react";
import { Card, CardContent } from "@/components/ui/card";

interface StatsCardProps {
  title: string;
  value: string | number;
  icon: ReactNode;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  variant?: "default" | "primary" | "success";
}

const StatsCard = ({ title, value, icon, trend, variant = "default" }: StatsCardProps) => {
  const variantStyles = {
    default: "bg-card",
    primary: "bg-gradient-to-br from-primary/10 to-primary/5 border-primary/20",
    success: "bg-gradient-to-br from-emerald-500/10 to-emerald-500/5 border-emerald-500/20",
  };

  return (
    <Card className={`${variantStyles[variant]} transition-all duration-300 hover:shadow-lg hover:-translate-y-0.5`}>
      <CardContent className="p-6">
        <div className="flex items-start justify-between">
          <div className="space-y-2">
            <p className="text-sm font-medium text-muted-foreground">{title}</p>
            <p className="text-3xl font-bold text-foreground">{value.toLocaleString()}</p>
            {trend && (
              <p className={`text-sm font-medium ${trend.isPositive ? "text-emerald-600" : "text-rose-600"}`}>
                {trend.isPositive ? "+" : "-"}{trend.value}% from last month
              </p>
            )}
          </div>
          <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center text-primary">
            {icon}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default StatsCard;
