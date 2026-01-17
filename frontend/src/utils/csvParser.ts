/**
 * Parser para o CSV de hospitais
 * Converte linhas CSV em objetos Hospital com coordenadas parseadas
 */

export interface HospitalData {
  period: string;
  periodFormatted: Date;
  region: string;
  institution: string;
  lat: number;
  lon: number;
  totalConsultations: number;
  firstConsultations: number;
  subsequentConsultations: number;
}

export function parseCSVLine(line: string): HospitalData | null {
  const parts = line.split(";");

  if (parts.length < 8) return null;

  // Parse coordenadas geográficas "38.529351, -8.881073"
  const coords = parts[4].split(",").map((s) => parseFloat(s.trim()));

  if (coords.length !== 2 || isNaN(coords[0]) || isNaN(coords[1])) {
    return null;
  }

  return {
    period: parts[0],
    periodFormatted: new Date(parts[1]),
    region: parts[2],
    institution: parts[3],
    lat: coords[0],
    lon: coords[1],
    totalConsultations: parseInt(parts[5]) || 0,
    firstConsultations: parseInt(parts[6]) || 0,
    subsequentConsultations: parseInt(parts[7]) || 0,
  };
}

export function parseCSV(csvText: string): HospitalData[] {
  const lines = csvText.trim().split("\n");
  const data: HospitalData[] = [];

  // Skip header
  for (let i = 1; i < lines.length; i++) {
    const parsed = parseCSVLine(lines[i]);
    if (parsed) {
      data.push(parsed);
    }
  }

  return data;
}

/**
 * Agrupa dados por hospital (soma consultas de todos os períodos)
 */
export function aggregateByHospital(data: HospitalData[]) {
  const hospitalMap = new Map<
    string,
    {
      name: string;
      region: string;
      lat: number;
      lon: number;
      totalConsultations: number;
    }
  >();

  data.forEach((row) => {
    const existing = hospitalMap.get(row.institution);

    if (existing) {
      existing.totalConsultations += row.totalConsultations;
    } else {
      hospitalMap.set(row.institution, {
        name: row.institution,
        region: row.region,
        lat: row.lat,
        lon: row.lon,
        totalConsultations: row.totalConsultations,
      });
    }
  });

  return Array.from(hospitalMap.values());
}

/**
 * Filtra dados por período específico
 */
export function filterByPeriod(data: HospitalData[], period: string) {
  return data.filter((row) => row.period === period);
}
