import { useMemo } from "react";
import { Pill, Building2, Calendar, TrendingUp } from "lucide-react";
import Layout from "@/components/Layout";
import StatsCard from "@/components/StatsCard";
import DistributionChart from "@/components/DistributionChart";
import TimelineChart from "@/components/TimelineChart";
import { useDistribution } from "@/context/DistributionContext";
import { hospitals } from "@/data/mockData";

const Dashboard = () => {
  const { distributions } = useDistribution();

  const stats = useMemo(() => {
    const totalDrugs = distributions.reduce((acc, d) => acc + d.totalDrugs, 0);
    const latestMonth = distributions[distributions.length - 1];
    const previousMonth = distributions[distributions.length - 2];
    
    const trend = previousMonth && latestMonth
      ? ((latestMonth.totalDrugs - previousMonth.totalDrugs) / previousMonth.totalDrugs) * 100
      : 0;

    return {
      totalDrugs,
      hospitalCount: hospitals.length,
      latestMonth: latestMonth?.month || "N/A",
      latestYear: latestMonth?.year || 2026,
      trend: Math.abs(Math.round(trend * 10) / 10),
      trendPositive: trend >= 0,
    };
  }, [distributions]);

  return (
    <Layout>
      <div className="p-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground">Dashboard</h1>
          <p className="text-muted-foreground mt-1">
            Overview of drug distribution across all hospitals
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <StatsCard
            title="Total Drugs Distributed"
            value={stats.totalDrugs}
            icon={<Pill className="w-6 h-6" />}
            variant="primary"
            trend={{ value: stats.trend, isPositive: stats.trendPositive }}
          />
          <StatsCard
            title="Active Hospitals"
            value={stats.hospitalCount}
            icon={<Building2 className="w-6 h-6" />}
          />
          <StatsCard
            title="Latest Distribution"
            value={`${stats.latestMonth} ${stats.latestYear}`}
            icon={<Calendar className="w-6 h-6" />}
          />
          <StatsCard
            title="Distribution Records"
            value={distributions.length}
            icon={<TrendingUp className="w-6 h-6" />}
            variant="success"
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <DistributionChart />
          <TimelineChart />
        </div>
      </div>
    </Layout>
  );
};

export default Dashboard;
