package org.project;

import weka.core.AttributeStats;
import weka.core.Instances;

/**
 * Explorer — Step 3 of the pipeline.
 *
 * Iterates over every attribute and prints descriptive statistics:
 *   • Numeric  → min, max, mean, stdDev
 *   • Nominal  → value frequencies
 */
public class Explorer {

    public static void explore(Instances data) {
        System.out.println("═══════════════════════════════════════════════════════");
        System.out.println("  STEP 3 ▸ EXPLORATORY DATA ANALYSIS");
        System.out.println("═══════════════════════════════════════════════════════");
        System.out.printf("  Dataset: %,d instances  ×  %d attributes%n%n",
                data.numInstances(), data.numAttributes());

        for (int i = 0; i < data.numAttributes(); i++) {
            AttributeStats stats = data.attributeStats(i);
            String name = data.attribute(i).name();
            boolean isNum = data.attribute(i).isNumeric();

            System.out.printf("  ┌─ [%d] %s  (%s)%n", i, name,
                    isNum ? "Numeric" : "Nominal");
            System.out.printf("  │  Distinct : %d%n", stats.distinctCount);
            System.out.printf("  │  Missing  : %d%n", stats.missingCount);

            if (isNum && stats.numericStats != null) {
                System.out.printf("  │  Min      : %.4f%n", stats.numericStats.min);
                System.out.printf("  │  Max      : %.4f%n", stats.numericStats.max);
                System.out.printf("  │  Mean     : %.4f%n", stats.numericStats.mean);
                System.out.printf("  │  StdDev   : %.4f%n", stats.numericStats.stdDev);
            } else if (data.attribute(i).isNominal()) {
                StringBuilder sb = new StringBuilder();
                for (int v = 0; v < data.attribute(i).numValues(); v++) {
                    if (v > 0) sb.append(", ");
                    sb.append(String.format("%s(%d)",
                            data.attribute(i).value(v), stats.nominalCounts[v]));
                }
                System.out.printf("  │  Values   : %s%n", sb);
            }

            System.out.println("  └───────────────────────────────────────────");
            System.out.println();
        }
    }
}
