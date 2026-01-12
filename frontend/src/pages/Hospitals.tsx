import { useState } from "react";
import { Building2, Edit2, Save, X } from "lucide-react";
import Layout from "@/components/Layout";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useToast } from "@/hooks/use-toast";
import { useDistribution } from "@/context/DistributionContext";
import { Hospital } from "@/data/mockData";

const Hospitals = () => {
  const { hospitals, setHospitals } = useDistribution();
  const { toast } = useToast();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValues, setEditValues] = useState<Hospital | null>(null);

  const handleEdit = (hospital: Hospital) => {
    setEditingId(hospital.id);
    setEditValues({ ...hospital });
  };

  const handleCancel = () => {
    setEditingId(null);
    setEditValues(null);
  };

  const handleSave = () => {
    if (!editValues) return;

    const totalAllocation = hospitals.reduce((acc, h) => {
      if (h.id === editValues.id) return acc + editValues.allocationPercentage;
      return acc + h.allocationPercentage;
    }, 0);

    if (totalAllocation !== 100) {
      toast({
        title: "Invalid Allocation",
        description: `Total allocation must equal 100%. Current total: ${totalAllocation}%`,
        variant: "destructive",
      });
      return;
    }

    const updatedHospitals = hospitals.map(h =>
      h.id === editValues.id ? editValues : h
    );
    setHospitals(updatedHospitals);
    setEditingId(null);
    setEditValues(null);

    toast({
      title: "Hospital Updated",
      description: `${editValues.name} allocation updated to ${editValues.allocationPercentage}%`,
    });
  };

  const totalAllocation = hospitals.reduce((acc, h) => acc + h.allocationPercentage, 0);

  return (
    <Layout>
      <div className="p-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground">Hospitais</h1>
          <p className="text-muted-foreground mt-1">
            Gerenciar a lista de hospitais e os percentuais de alocação.
          </p>
        </div>

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Building2 className="w-5 h-5" />
                  Alocações hospitalares
                </CardTitle>
              </div>
              <div className={`px-4 py-2 rounded-lg font-medium ${
                totalAllocation === 100
                  ? "bg-emerald-500/10 text-emerald-600"
                  : "bg-rose-500/10 text-rose-600"
              }`}>
                Total: {totalAllocation}%
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-12">#</TableHead>
                  <TableHead>Hospital Name</TableHead>
                  <TableHead className="text-right w-40">Allocation %</TableHead>
                  <TableHead className="text-right w-24">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {hospitals.map((hospital, index) => (
                  <TableRow key={hospital.id}>
                    <TableCell className="text-muted-foreground">{index + 1}</TableCell>
                    <TableCell className="font-medium">
                      {editingId === hospital.id ? (
                        <Input
                          value={editValues?.name || ""}
                          onChange={(e) => setEditValues(prev => prev ? { ...prev, name: e.target.value } : null)}
                        />
                      ) : (
                        hospital.name
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {editingId === hospital.id ? (
                        <Input
                          type="number"
                          min="0"
                          max="100"
                          className="w-24 ml-auto text-right"
                          value={editValues?.allocationPercentage || 0}
                          onChange={(e) => setEditValues(prev => prev ? { ...prev, allocationPercentage: parseInt(e.target.value) || 0 } : null)}
                        />
                      ) : (
                        <span className="font-semibold">{hospital.allocationPercentage}%</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {editingId === hospital.id ? (
                        <div className="flex justify-end gap-1">
                          <Button size="icon" variant="ghost" onClick={handleSave}>
                            <Save className="w-4 h-4" />
                          </Button>
                          <Button size="icon" variant="ghost" onClick={handleCancel}>
                            <X className="w-4 h-4" />
                          </Button>
                        </div>
                      ) : (
                        <Button size="icon" variant="ghost" onClick={() => handleEdit(hospital)}>
                          <Edit2 className="w-4 h-4" />
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
};

export default Hospitals;
