import { useEffect, useState } from "react";
import Layout from "@/components/Layout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { apiService, ModelMetrics } from "@/services/api";
import { Loader2, TrendingUp, Activity, CheckCircle2, AlertCircle } from "lucide-react";

const ModelMetricsPage = () => {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.getModelMetrics();
      setMetrics(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erro ao carregar métricas");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  const getEfficiencyColor = (efficiency: number) => {
    if (efficiency >= 80) return "text-green-600";
    if (efficiency >= 60) return "text-yellow-600";
    return "text-red-600";
  };

  const getEfficiencyBadge = (efficiency: number) => {
    if (efficiency >= 80) return <Badge className="bg-green-600">Excelente</Badge>;
    if (efficiency >= 60) return <Badge className="bg-yellow-600">Bom</Badge>;
    return <Badge className="bg-red-600">Necessita Melhoria</Badge>;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-2">Calculando métricas...</span>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>{error}</AlertDescription>
        <Button onClick={fetchMetrics} variant="outline" className="mt-2">
          Tentar Novamente
        </Button>
      </Alert>
    );
  }

  if (!metrics) {
    return (
      <Alert>
        <AlertDescription>Nenhuma métrica disponível</AlertDescription>
      </Alert>
    );
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold">Eficácia do Modelo</h1>
          </div>
          <Button onClick={fetchMetrics} variant="outline">
            <Activity className="mr-2 h-4 w-4" />
            Atualizar
          </Button>
        </div>

        {/* Card Principal - Eficiência */}
        <Card className="border-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Eficiência Global
            </CardTitle>
            <CardDescription>
              Baseada no coeficiente de determinação (R²) no conjunto de teste
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                <div className={`text-6xl font-bold ${getEfficiencyColor(metrics.metrics.efficiency_percentage)}`}>
                  {metrics.metrics.efficiency_percentage.toFixed(1)}%
                </div>
                <p className="text-sm text-muted-foreground mt-2">
                  Modelo: {metrics.model_type}
                </p>
              </div>
              <div className="text-right">
                {getEfficiencyBadge(metrics.metrics.efficiency_percentage)}
                <div className="mt-4 text-sm text-muted-foreground">
                  <div>Treino: {metrics.training_samples} amostras</div>
                  <div>Teste: {metrics.test_samples} amostras</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Métricas Detalhadas */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">R² Treino</CardTitle>
              <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{(metrics.metrics.r2_train * 100).toFixed(1)}%</div>
              <p className="text-xs text-muted-foreground">Ajuste aos dados de treino</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">R² Teste</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{(metrics.metrics.r2_test * 100).toFixed(1)}%</div>
              <p className="text-xs text-muted-foreground">Capacidade de generalização</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">MAE</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{metrics.metrics.mae_test.toFixed(1)}</div>
              <p className="text-xs text-muted-foreground">Erro médio absoluto</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">RMSE</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{metrics.metrics.rmse_test.toFixed(1)}</div>
              <p className="text-xs text-muted-foreground">Raiz do erro quadrático médio</p>
            </CardContent>
          </Card>
        </div>

        {/* Comparação de Modelos */}
        <Card>
          <CardHeader>
            <CardTitle>Comparação de Modelos</CardTitle>
            <CardDescription>
              Performance de diferentes algoritmos testados
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {Object.entries(metrics.model_comparison).map(([name, scores]) => (
                <div key={name} className="flex items-center justify-between border-b pb-2 last:border-b-0">
                  <div className="font-medium">{name}</div>
                  <div className="flex gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Treino:</span>{" "}
                      <span className="font-semibold">{(scores.r2_train * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Teste:</span>{" "}
                      <span className="font-semibold">{(scores.r2_test * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        <Card>
            <CardHeader>
        <CardDescription>
        <p className="text-muted-foreground">
            Avaliação de performance do modelo de previsão
        </p>
        </CardDescription>
        </CardHeader>
        {/* Imagem do treinamento */}
        <div className="flex justify-center">
          <img
            src="http://localhost:8000/model/training-image"
            alt="Resultados do Treinamento"
            className="rounded-lg border shadow-lg max-w-4xl w-full bg-white"
            style={{ background: 'white' }}
          />
        </div>
        </Card>

        {/* Interpretação das Métricas */}
        <Card>
          <CardHeader>
            <CardTitle>Interpretação das Métricas</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div>
              <h4 className="font-semibold mb-1">R² (Coeficiente de Determinação)</h4>
              <p className="text-sm text-muted-foreground">
                Mede a proporção da variância explicada pelo modelo. Valores próximos de 100% indicam
                que o modelo explica bem os dados. Um R² de teste alto indica boa capacidade de generalização.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-1">MAE (Mean Absolute Error)</h4>
              <p className="text-sm text-muted-foreground">
                Erro médio absoluto entre predições e valores reais. Quanto menor, melhor. Representa
                o desvio médio em unidades de medicamentos.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-1">RMSE (Root Mean Squared Error)</h4>
              <p className="text-sm text-muted-foreground">
                Similar ao MAE mas penaliza mais erros grandes. Útil para identificar predições
                muito distantes dos valores reais.
              </p>
            </div>
          </CardContent>
        </Card>

        <div className="text-xs text-muted-foreground text-center">
          Última atualização: {new Date(metrics.timestamp).toLocaleString('pt-PT')}
        </div>
      </div>
    </Layout>
  );
};

export default ModelMetricsPage;
