import { useState, useMemo } from "react";
import { Calculator, Check, Calendar } from "lucide-react";
import Layout from "@/components/Layout";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Checkbox } from "@/components/ui/checkbox";
import { useToast } from "@/hooks/use-toast";
import { useDistribution } from "@/context/DistributionContext";
import { months, generateDistributionData } from "@/data/mockData";

type DistributionMode = "month" | "quarter" | "year";

const Distribute = () => {
  const { hospitals, addDistribution, selectedMonth, setSelectedYear, selectedYear } = useDistribution();
  const { toast } = useToast();

  const [mode, setMode] = useState<DistributionMode>("month");
  const [selectedMonths, setSelectedMonths] = useState<string[]>([selectedMonth]);
  const [selectedQuarter, setSelectedQuarter] = useState<string>("1");
  const [isCalculated, setIsCalculated] = useState(false);

  // Default drug pool for calculation (since input was removed)
  const totalDrugsPool = 50000;
  const years = [2024, 2025, 2026, 2027];

  // Determine active months based on UI selection
  const activeMonths = useMemo(() => {
    if (mode === "month") return selectedMonths;
    if (mode === "year") return months;
    if (mode === "quarter") {
      const q = parseInt(selectedQuarter);
      return months.slice((q - 1) * 3, q * 3);
    }
    return [];
  }, [mode, selectedMonths, selectedQuarter]);

  const distributionPreview = useMemo(() => {
    // Generates preview based on the fixed drug pool and active months
    return generateDistributionData(totalDrugsPool, hospitals);
  }, [hospitals]);

  const handleMonthToggle = (month: string) => {
    setSelectedMonths(prev =>
      prev.includes(month) ? prev.filter(m => m !== month) : [...prev, month]
    );
    setIsCalculated(false);
  };

  const handleCalculate = () => {
    if (activeMonths.length === 0) {
      toast({
        title: "Selection Required",
        description: "Please select at least one month for distribution.",
        variant: "destructive",
      });
      return;
    }
    setIsCalculated(true);
    toast({
      title: "Preview Generated",
      description: `Distribution calculated for ${activeMonths.length} month(s).`,
    });
  };

  const handleSave = () => {
    activeMonths.forEach(month => {
      addDistribution({
        month,
        year: selectedYear,
        totalDrugs: totalDrugsPool,
        distributions: distributionPreview.map(d => ({
          hospitalId: d.hospitalId,
          quantity: d.quantity,
        })),
      });
    });

    toast({
      title: "Success",
      description: `Distribution saved for ${activeMonths.length} month(s) in ${selectedYear}.`,
    });
  };

  return (
    <Layout>
      <div className="p-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold">Calcular a distribuição</h1>
          <p className="text-muted-foreground mt-1">Selecione o seu período de distribuição.</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-1">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Calendar className="w-5 h-5" />
                Seleção do período de tempo
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <Tabs value={mode} onValueChange={(v) => {
                setMode(v as DistributionMode);
                setIsCalculated(false);
              }}>
                <TabsList className="grid grid-cols-3 w-full">
                  <TabsTrigger value="month">Meses</TabsTrigger>
                  <TabsTrigger value="quarter">Trimestre</TabsTrigger>
                  <TabsTrigger value="year">Ano inteiro</TabsTrigger>
                </TabsList>
              </Tabs>

              <div className="space-y-2">
                <Label>Ano</Label>
                <Select value={selectedYear.toString()} onValueChange={(v) => setSelectedYear(parseInt(v))}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>{years.map(y => <SelectItem key={y} value={y.toString()}>{y}</SelectItem>)}</SelectContent>
                </Select>
              </div>

              {mode === "month" && (
                <div className="space-y-3">
                  <Label>Selecione o(s) mês(es)</Label>
                  <div className="grid grid-cols-2 gap-2 border p-3 rounded-md max-h-48 overflow-y-auto bg-card">
                    {months.map(m => (
                      <div key={m} className="flex items-center space-x-2">
                        <Checkbox id={m} checked={selectedMonths.includes(m)} onCheckedChange={() => handleMonthToggle(m)} />
                        <label htmlFor={m} className="text-sm cursor-pointer">{m}</label>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {mode === "quarter" && (
                <div className="space-y-2">
                  <Label>Selecione o trimestre</Label>
                  <Select value={selectedQuarter} onValueChange={(v) => { setSelectedQuarter(v); setIsCalculated(false); }}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">Q1 (Jan, Fev, Mar)</SelectItem>
                      <SelectItem value="2">Q2 (Abr, Mai, Jun)</SelectItem>
                      <SelectItem value="3">Q3 (Jul, Ago, Set)</SelectItem>
                      <SelectItem value="4">Q4 (Out, Nov, Dec)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              )}

              <div className="pt-4 space-y-3">
                <Button onClick={handleCalculate} className="w-full" size="lg">
                  <Calculator className="w-4 h-4 mr-2" /> Distribuição prévia
                </Button>
                {isCalculated && (
                  <Button onClick={handleSave} variant="secondary" className="w-full" size="lg">
                    <Check className="w-4 h-4 mr-2" /> Salvar em {activeMonths.length} Meses
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>

          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle>Distribuição prévia</CardTitle>
              <CardDescription>
                {activeMonths.length > 0
                  ? `Allocation for: ${activeMonths.join(", ")}`
                  : "Select a timeframe to see the preview"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Hospital</TableHead>
                    <TableHead className="text-right">Alocação %</TableHead>
                    <TableHead className="text-right">Quantidade</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {distributionPreview.map((item) => (
                    <TableRow key={item.hospitalId}>
                      <TableCell className="font-medium">{item.hospitalName}</TableCell>
                      <TableCell className="text-right">{item.percentage}%</TableCell>
                      <TableCell className="text-right font-semibold">
                        {item.quantity.toLocaleString()}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </div>
      </div>
    </Layout>
  );
};

export default Distribute;