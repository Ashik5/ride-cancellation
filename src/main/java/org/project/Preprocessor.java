package org.project;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 * Preprocessor — Step 2 of the pipeline.
 *
 * 1. Drops data-leakage columns  (cancellation_reason)
 * 2. Drops non-predictive columns (IDs, raw timestamps)
 * 3. Imputes missing values       (mode for nominal, mean for numeric)
 * 4. Sets the class attribute      (booking_status)
 */
public class Preprocessor {

    /* ── Columns to drop ──────────────────────────────────────── */

    /** Data-leakage: known only AFTER cancellation */
    private static final String[] LEAKAGE_COLUMNS = {
        "cancellation_reason", "cancellation reason",
        "Cancellation_Reason", "Cancellation Reason",
        "cancel_reason", "Cancel_Reason",
        // Actual CSV column names
        "Reason for cancelling by Customer",
        "Cancelled Rides by Customer",
        "Driver Cancellation Reason",
        "Cancelled Rides by Driver",
        "Incomplete Rides",
        "Incomplete Rides Reason"
    };

    /** Non-predictive identifiers and raw timestamps */
    private static final String[] DROP_COLUMNS = {
        "booking_id", "Booking_ID", "ride_id", "Ride_ID", "id", "ID",
        "booking_time", "booking time", "Booking_Time", "Booking Time",
        "timestamp", "Timestamp", "date", "Date",
        "booking_datetime", "ride_date",
        // Actual CSV column names
        "Booking ID", "Customer ID", "Time",
        // High-cardinality locations (100s of unique values → OOM)
        "Pickup Location", "Drop Location"
    };

    /** Candidate names for the target / class attribute */
    private static final String[] CLASS_CANDIDATES = {
        "booking_status", "booking status",
        "Booking_Status", "Booking Status",
        "Booking Status ",                 // trailing space in some CSVs
        "status", "cancelled", "is_cancelled"
    };

    /* ── Public API ───────────────────────────────────────────── */

    /**
     * Full preprocessing pipeline.
     *
     * @param raw the raw Instances straight from the CSV loader
     * @return clean, class-indexed Instances ready for modelling
     */
    public static Instances preprocess(Instances raw) throws Exception {
        System.out.println("═══════════════════════════════════════════════════════");
        System.out.println("  STEP 2 ▸ DATA PREPROCESSING");
        System.out.println("═══════════════════════════════════════════════════════");

        Instances data = new Instances(raw);

        // 1. Remove leakage columns (CRITICAL)
        data = removeByName(data, LEAKAGE_COLUMNS, "⚠ LEAKAGE");

        // 2. Remove ID / timestamp columns
        data = removeByName(data, DROP_COLUMNS, "ID/timestamp");

        // 3. Impute missing values
        data = imputeMissing(data);

        // 4. Set class index
        data = resolveClassIndex(data);

        System.out.printf("  ✓ Final shape : %,d instances × %d attributes%n",
                data.numInstances(), data.numAttributes());
        System.out.printf("  ✓ Class attr  : '%s' (index %d, %d values)%n",
                data.classAttribute().name(),
                data.classIndex(),
                data.classAttribute().numValues());
        System.out.println();

        return data;
    }

    /* ── Internals ────────────────────────────────────────────── */

    private static Instances removeByName(Instances data,
                                          String[] names,
                                          String tag) throws Exception {
        StringBuilder idxBuf = new StringBuilder();
        StringBuilder logBuf = new StringBuilder();

        for (String name : names) {
            int idx = findAttr(data, name);
            if (idx >= 0) {
                if (idxBuf.length() > 0) idxBuf.append(",");
                idxBuf.append(idx + 1);                     // WEKA is 1-based
                if (logBuf.length() > 0) logBuf.append(", ");
                logBuf.append(data.attribute(idx).name());
            }
        }

        if (idxBuf.length() > 0) {
            Remove rm = new Remove();
            rm.setAttributeIndices(idxBuf.toString());
            rm.setInputFormat(data);
            data = Filter.useFilter(data, rm);
            System.out.printf("  ✓ Dropped [%s]: %s%n", tag, logBuf);
        }

        return data;
    }

    private static int findAttr(Instances data, String name) {
        for (int i = 0; i < data.numAttributes(); i++) {
            if (data.attribute(i).name().trim().equalsIgnoreCase(name.trim())) {
                return i;
            }
        }
        return -1;
    }

    private static Instances imputeMissing(Instances data) throws Exception {
        long missing = 0;
        for (int i = 0; i < data.numInstances(); i++)
            for (int j = 0; j < data.numAttributes(); j++)
                if (data.instance(i).isMissing(j)) missing++;

        if (missing > 0) {
            ReplaceMissingValues rmv = new ReplaceMissingValues();
            rmv.setInputFormat(data);
            data = Filter.useFilter(data, rmv);
            System.out.printf("  ✓ Imputed %,d missing values (mean/mode)%n", missing);
        } else {
            System.out.println("  ✓ No missing values detected");
        }
        return data;
    }

    private static Instances resolveClassIndex(Instances data) {
        for (String candidate : CLASS_CANDIDATES) {
            int idx = findAttr(data, candidate);
            if (idx >= 0) {
                data.setClassIndex(idx);
                return data;
            }
        }
        // Fallback: use last attribute
        data.setClassIndex(data.numAttributes() - 1);
        System.out.println("  ⚠ Class not found by name — defaulting to last attribute");
        return data;
    }
}
