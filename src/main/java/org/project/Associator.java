package org.project;

import weka.associations.Apriori;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

/**
 * Associator — Step 6: Association Rule Mining.
 * Applies Discretize filter (numeric → nominal bins) then runs Apriori.
 */
public class Associator {

    public static void run(Instances data) throws Exception {
        System.out.println("═══════════════════════════════════════════════════════");
        System.out.println("  STEP 6 ▸ ASSOCIATION RULE MINING  (Apriori)");
        System.out.println("═══════════════════════════════════════════════════════");

        // Apriori requires all-nominal data → discretize numeric attributes
        Discretize disc = new Discretize();
        disc.setInputFormat(data);
        Instances discretized = Filter.useFilter(data, disc);
        System.out.println("  ✓ Applied Discretize filter to numeric attributes");

        Apriori apriori = new Apriori();
        apriori.setNumRules(10);
        apriori.buildAssociations(discretized);

        System.out.println("  ✓ Apriori completed\n");
        System.out.println("  ▸ Top Association Rules:");
        System.out.println("  ─────────────────────────────────────────────");
        System.out.println(apriori);
        System.out.println();
    }
}
