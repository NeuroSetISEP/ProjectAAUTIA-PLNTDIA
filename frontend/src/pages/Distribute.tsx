import { useState, useMemo } from "react";
import { Calculator, Check, Calendar, Wifi, WifiOff, Crown } from "lucide-react";
import Layout from "@/components/Layout";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { useDistribution } from "@/context/DistributionContext";
import { months, generateDistributionData } from "@/data/mockData";
import { apiService } from "@/services/api";

type DistributionMode = "month" | "quarter" | "year";
type SortKey = "institution" | "region" | "predicted_consumption" | "allocated_amount" | "priority_weight" | "coverage" | "safety_stock";

const Distribute = () => {
  const { hospitals, addDistribution, selectedMonth, setSelectedYear, selectedYear, isBackendConnected } = useDistribution();
  const { toast } = useToast();

  const [mode, setMode] = useState<DistributionMode>("month");
  const [selectedMonths, setSelectedMonths] = useState<string[]>([selectedMonth]);
  const [selectedQuarter, setSelectedQuarter] = useState<string>("1");
  const [availableStock, setAvailableStock] = useState<number>(35000);
  const [isCalculating, setIsCalculating] = useState(false);
  const [distributionResults, setDistributionResults] = useState<any[]>([]);
  const [isCalculated, setIsCalculated] = useState(false);
  const [estimatedTotal, setEstimatedTotal] = useState<number>(0);
  const [sortKey, setSortKey] = useState<SortKey>("institution");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");

  // Default drug pool for calculation (fallback for mock mode)
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

  // Função para ordenar hospitais
  const getSortedHospitals = (hospitals: any[]) => {
    if (!hospitals) return [];

    return [...hospitals].sort((a, b) => {
      let aValue: any = a[sortKey];
      let bValue: any = b[sortKey];

      // Cobertura é calculada
      if (sortKey === "coverage") {
        aValue = a.predicted_consumption > 0 ? a.allocated_amount / a.predicted_consumption : 0;
        bValue = b.predicted_consumption > 0 ? b.allocated_amount / b.predicted_consumption : 0;
      }

      if (typeof aValue === "string") {
        return sortOrder === "asc"
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }

      return sortOrder === "asc" ? aValue - bValue : bValue - aValue;
    });
  };

  // Handler para clicar no cabeçalho
  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortOrder(key === "institution" || key === "region" ? "asc" : "desc");
    }
  };

  const handleMonthToggle = (month: string) => {
    setSelectedMonths(prev =>
      prev.includes(month) ? prev.filter(m => m !== month) : [...prev, month]
    );
    setIsCalculated(false);
    setDistributionResults([]);
  };

  // Função auxiliar para converter nome do mês para número
  const monthNameToNumber = (monthName: string): number => {
    const monthNames = ["January", "February", "March", "April", "May", "June",
      "July", "August", "September", "October", "November", "December"];
    return monthNames.indexOf(monthName) + 1;
  };

  // Função auxiliar para converter número do mês para nome
  const monthNumberToName = (monthNumber: number): string => {
    const monthNames = ["January", "February", "March", "April", "May", "June",
      "July", "August", "September", "October", "November", "December"];
    return monthNames[monthNumber - 1] || "Unknown";
  };

  const handleCalculate = async () => {
    if (activeMonths.length === 0) {
      toast({
        title: "Selection Required",
        description: "Please select at least one month for distribution.",
        variant: "destructive",
      });
      return;
    }

    setIsCalculating(true);
    setDistributionResults([]);

    try {
      if (isBackendConnected) {
        // Usar API real
        const monthNumbers = activeMonths.map(month => monthNameToNumber(month));

        const request = {
          months: monthNumbers,
          year: selectedYear,
          available_stock: availableStock,
          mode: mode
        };

        const results = await apiService.optimizeDistribution(request);
        setDistributionResults(results);

        // Atualizar estimativa total baseado nos resultados
        if (results.length > 0) {
          const totalPredicted = results.reduce((sum, r) => sum + r.total_predicted, 0);
          setEstimatedTotal(totalPredicted);
        }

        toast({
          title: "🤖 AI Optimization Complete",
          description: `Optimized distribution for ${results.length} period(s) using ML predictions.`,
        });
      } else {
        // Usar dados mock
        const mockResults = activeMonths.map(month => {
          const monthNum = monthNameToNumber(month);
          const stockPercentage = Math.min(100, Math.round((availableStock / totalDrugsPool) * 100));

          // Criar hospitais mock com IDs consistentes
          const mockHospitals = distributionPreview.map((h: any, index: number) => {
            const predictedConsumption = Math.round(Math.random() * 2000 + 500);
            const allocatedAmount = Math.round(h.amount * (stockPercentage / 100));
            const safetyStock = Math.round(predictedConsumption * 0.05); // 5% mock safety stock

            return {
              id: h.hospitalId || `hospital-${index}`,
              institution: h.name,
              region: h.region || "Mock Region",
              predicted_consumption: predictedConsumption,
              priority_weight: Math.random(),
              allocated_amount: allocatedAmount,
              safety_stock: safetyStock,
              is_key_hospital: index % 5 === 0 // Mock: every 5th hospital is a Key Hospital
            };
          });

          const totalPredicted = mockHospitals.reduce((sum: number, h: any) => sum + h.predicted_consumption, 0);

          return {
            period: `${selectedYear}-${monthNum.toString().padStart(2, '0')}`,
            year: selectedYear,
            hospitals: mockHospitals,
            total_predicted: totalPredicted,
            available_stock: availableStock * (stockPercentage / 100),
            optimization_score: 0.85 + Math.random() * 0.1
          };
        });

        setDistributionResults(mockResults);
        setEstimatedTotal(mockResults.reduce((sum, r) => sum + r.total_predicted, 0));

        toast({
          title: "📊 Mock Distribution Generated",
          description: `Generated distribution preview for ${mockResults.length} period(s).`,
        });
      }

      setIsCalculated(true);
    } catch (error) {
      toast({
        title: "Calculation Failed",
        description: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        variant: "destructive",
      });
    } finally {
      setIsCalculating(false);
    }
  };

  const handleSave = () => {
    distributionResults.forEach(result => {
      const monthNum = parseInt(result.period.split('-')[1]);
      const monthName = monthNumberToName(monthNum);

      addDistribution({
        month: monthName,
        year: result.year,
        totalDrugs: result.available_stock,
        distributions: result.hospitals.map((h: any) => ({
          hospitalId: h.id,
          quantity: h.allocated_amount,
        })),
      });
    });

    toast({
      title: "Success",
      description: `Distribution saved for ${distributionResults.length} period(s) in ${selectedYear}.`,
    });
  };

  return (
    <Layout>
      <div className="p-8">
        <div className="mb-8 flex items-center justify-between">
          <div className="flex items-center gap-4">

            <div>
              <h1 className="text-3xl font-bold">Calcular a distribuição</h1>
              <p className="text-muted-foreground mt-1">Selecione o seu período de distribuição.</p>
            </div>
          </div>
          <Badge variant={isBackendConnected ? "default" : "destructive"} className="flex items-center gap-2">
            {isBackendConnected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
            {isBackendConnected ? "AI Backend Connected" : "Mock Mode"}
          </Badge>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-1">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Calendar className="w-5 h-5" />
                Configuração
              </CardTitle>
              <CardDescription>
                {isBackendConnected ? "Configure os parâmetros para otimização AI" : "Configuração para simulação"}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <Tabs value={mode} onValueChange={(v) => {
                setMode(v as DistributionMode);
                setIsCalculated(false);
                setDistributionResults([]);
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
                  <SelectContent>
                    {years.map(y => (
                      <SelectItem key={y} value={y.toString()}>{y}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="available-stock">
                  Carbapenemes Disponíveis
                </Label>
                <Input
                  id="available-stock"
                  type="number"
                  min="1000"
                  max="100000"
                  value={availableStock}
                  onChange={(e) => {
                    setAvailableStock(parseInt(e.target.value) || 35000);
                    setIsCalculated(false);
                    setDistributionResults([]);
                  }}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground">
                  Quantidade total de carbapenemes disponíveis para distribuição
                </p>
              </div>

              {mode === "month" && (
                <div className="space-y-3">
                  <Label>Selecione o(s) mês(es)</Label>
                  <div className="grid grid-cols-2 gap-2 border p-3 rounded-md max-h-48 overflow-y-auto bg-card">
                    {months.map(m => (
                      <div key={m} className="flex items-center space-x-2">
                        <Checkbox
                          id={m}
                          checked={selectedMonths.includes(m)}
                          onCheckedChange={() => handleMonthToggle(m)}
                        />
                        <label htmlFor={m} className="text-sm cursor-pointer">{m}</label>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {mode === "quarter" && (
                <div className="space-y-2">
                  <Label>Trimestre</Label>
                  <Select value={selectedQuarter} onValueChange={setSelectedQuarter}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">Q1 (Jan - Mar)</SelectItem>
                      <SelectItem value="2">Q2 (Apr - Jun)</SelectItem>
                      <SelectItem value="3">Q3 (Jul - Sep)</SelectItem>
                      <SelectItem value="4">Q4 (Oct - Dec)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              )}

              <Button
                onClick={handleCalculate}
                className="w-full"
                disabled={isCalculating || activeMonths.length === 0}
              >
                <Calculator className="w-4 h-4 mr-2" />
                {isCalculating ? "Calculando..." : (isBackendConnected ? "Otimizar com AI" : "Gerar Preview")}
              </Button>

              {isCalculated && distributionResults.length > 0 && (
                <Button onClick={handleSave} variant="outline" className="w-full">
                  <Check className="w-4 h-4 mr-2" />
                  Salvar Distribuição
                </Button>
              )}
            </CardContent>
          </Card>

          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Resultados da Distribuição</span>
                {distributionResults.length > 0 && (
                  <Badge variant="outline">
                    {distributionResults.length} período(s) calculado(s)
                  </Badge>
                )}
              </CardTitle>
              <CardDescription>
                {isBackendConnected
                  ? "Distribuição otimizada baseada em Previsões ML e algoritmo genético"
                  : "Preview baseado em dados simulados"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {distributionResults.length > 0 ? (
                <div className="space-y-6">
                  {distributionResults.map((result, index) => (
                    <div key={index} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold">
                          {monthNumberToName(parseInt(result.period.split('-')[1]))} {result.year}
                        </h3>
                        <div className="flex gap-2">
                          <Badge variant="outline">
                            {result.available_stock.toLocaleString()} disponível
                          </Badge>
                        </div>
                      </div>

                      <div className="overflow-x-auto">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead
                                onClick={() => handleSort("institution")}
                                className="cursor-pointer select-none hover:bg-accent"
                              >
                                Hospital {sortKey === "institution" && (sortOrder === "asc" ? "▲" : "▼")}
                              </TableHead>
                              <TableHead
                                onClick={() => handleSort("region")}
                                className="cursor-pointer select-none hover:bg-accent"
                              >
                                Região {sortKey === "region" && (sortOrder === "asc" ? "▲" : "▼")}
                              </TableHead>
                              <TableHead
                                onClick={() => handleSort("predicted_consumption")}
                                className="cursor-pointer select-none hover:bg-accent"
                              >
                                Consumo Previsto {sortKey === "predicted_consumption" && (sortOrder === "asc" ? "▲" : "▼")}
                              </TableHead>
                              <TableHead
                                onClick={() => handleSort("safety_stock")}
                                className="cursor-pointer select-none hover:bg-accent"
                              >
                                Safety Stock {sortKey === "safety_stock" && (sortOrder === "asc" ? "▲" : "▼")}
                              </TableHead>
                              <TableHead
                                onClick={() => handleSort("allocated_amount")}
                                className="cursor-pointer select-none hover:bg-accent"
                              >
                                Alocação {sortKey === "allocated_amount" && (sortOrder === "asc" ? "▲" : "▼")}
                              </TableHead>
                              <TableHead
                                onClick={() => handleSort("priority_weight")}
                                className="cursor-pointer select-none hover:bg-accent"
                              >
                                Prioridade {sortKey === "priority_weight" && (sortOrder === "asc" ? "▲" : "▼")}
                              </TableHead>
                              <TableHead
                                onClick={() => handleSort("coverage")}
                                className="cursor-pointer select-none hover:bg-accent"
                              >
                                % Cobertura {sortKey === "coverage" && (sortOrder === "asc" ? "▲" : "▼")}
                              </TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {getSortedHospitals(result.hospitals || []).map((hospital: any, i: number) => {
                              const coverage = hospital.predicted_consumption > 0
                                ? (hospital.allocated_amount / hospital.predicted_consumption)
                                : 0;

                              return (
                                <TableRow key={hospital.id || i}>
                                  <TableCell className="font-medium">
                                    <div className="flex items-center gap-2">
                                      {hospital.institution}
                                      {hospital.is_key_hospital && (
                                        <div title="Hospital Chave da Região (Prioridade Máxima)">
                                          <Crown className="w-4 h-4 text-yellow-500 fill-yellow-500" />
                                        </div>
                                      )}
                                    </div>
                                  </TableCell>
                                  <TableCell>{hospital.region}</TableCell>
                                  <TableCell>{hospital.predicted_consumption.toFixed(0)}</TableCell>
                                  <TableCell className="text-muted-foreground">
                                    {hospital.safety_stock}
                                  </TableCell>
                                  <TableCell
                                    className={`font-semibold ${hospital.allocated_amount < hospital.safety_stock ? "text-red-500" : ""}`}
                                    title={hospital.allocated_amount < hospital.safety_stock ? "Below Safety Stock" : undefined}
                                  >
                                    {hospital.allocated_amount.toLocaleString()}
                                  </TableCell>
                                  <TableCell>
                                    <Badge variant="outline">
                                      {(hospital.priority_weight * 100).toFixed(1)}%
                                    </Badge>
                                  </TableCell>
                                  <TableCell>
                                    {hospital.predicted_consumption > 0 ? (
                                      <Badge variant={
                                        coverage >= 0.9
                                          ? "default"
                                          : coverage >= 0.7
                                          ? "secondary"
                                          : "destructive"
                                      }>
                                        {(coverage * 100).toFixed(0)}%
                                      </Badge>
                                    ) : (
                                      <Badge variant="outline">N/A</Badge>
                                    )}
                                  </TableCell>
                                </TableRow>
                              );
                            })}
                          </TableBody>
                        </Table>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12 text-muted-foreground">
                  <Calculator className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Selecione o período e clique em calcular para ver a distribuição otimizada.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </Layout>
  );
};

export default Distribute;