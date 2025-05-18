package org.example;

import java.io.*;
import java.util.*;

public class HousingDataLoader {
    // Define class to store the housing data
    public static class HousingData {
        // Features
        private double price;
        private double area;
        private int bedrooms;
        private int bathrooms;
        private int stories;
        private boolean mainroad;
        private boolean guestroom;
        private boolean basement;
        private boolean hotwaterheating;
        private boolean airconditioning;
        private int parking;
        private boolean prefarea;
        private String furnishingstatus;

        // Normalized features for ANN (will be set after normalization)
        private double[] normalizedFeatures;
        private double[] normalizedTarget;

        public HousingData(String[] values) {
            this.price = Double.parseDouble(values[0]);
            this.area = Double.parseDouble(values[1]);
            this.bedrooms = Integer.parseInt(values[2]);
            this.bathrooms = Integer.parseInt(values[3]);
            this.stories = Integer.parseInt(values[4]);
            this.mainroad = values[5].equalsIgnoreCase("yes");
            this.guestroom = values[6].equalsIgnoreCase("yes");
            this.basement = values[7].equalsIgnoreCase("yes");
            this.hotwaterheating = values[8].equalsIgnoreCase("yes");
            this.airconditioning = values[9].equalsIgnoreCase("yes");
            this.parking = Integer.parseInt(values[10]);
            this.prefarea = values[11].equalsIgnoreCase("yes");
            this.furnishingstatus = values[12];
        }

        public double getPrice() {
            return price;
        }

        public double getArea() {
            return area;
        }

        public double[] getNormalizedFeatures() {
            return normalizedFeatures;
        }

        public void setNormalizedFeatures(double[] normalizedFeatures) {
            this.normalizedFeatures = normalizedFeatures;
        }

        public double[] getNormalizedTarget() {
            return normalizedTarget;
        }

        public void setNormalizedTarget(double[] normalizedTarget) {
            this.normalizedTarget = normalizedTarget;
        }

        // Returns an array of all features (non-normalized)
        public double[] getRawFeatures() {
            double[] features = new double[12];
            features[0] = area;
            features[1] = bedrooms;
            features[2] = bathrooms;
            features[3] = stories;
            features[4] = mainroad ? 1.0 : 0.0;
            features[5] = guestroom ? 1.0 : 0.0;
            features[6] = basement ? 1.0 : 0.0;
            features[7] = hotwaterheating ? 1.0 : 0.0;
            features[8] = airconditioning ? 1.0 : 0.0;
            features[9] = parking;
            features[10] = prefarea ? 1.0 : 0.0;

            // Handle furnishing status with one-hot encoding
            if (furnishingstatus.equals("furnished")) {
                features[11] = 0.0;
            } else if (furnishingstatus.equals("semi-furnished")) {
                features[11] = 1.0;
            } else { // unfurnished
                features[11] = 2.0;
            }

            return features;
        }

        @Override
        public String toString() {
            return "HousingData{" +
                    "price=" + price +
                    ", area=" + area +
                    ", bedrooms=" + bedrooms +
                    ", bathrooms=" + bathrooms +
                    ", stories=" + stories +
                    ", mainroad=" + mainroad +
                    ", guestroom=" + guestroom +
                    ", basement=" + basement +
                    ", hotwaterheating=" + hotwaterheating +
                    ", airconditioning=" + airconditioning +
                    ", parking=" + parking +
                    ", prefarea=" + prefarea +
                    ", furnishingstatus='" + furnishingstatus + '\'' +
                    '}';
        }
    }

    private List<HousingData> housingDataList;
    private double[] minFeatures;
    private double[] maxFeatures;
    private double minPrice;
    private double maxPrice;

    public HousingDataLoader() {
        housingDataList = new ArrayList<>();
    }

    // Load data from CSV file
    public void loadData(String filename) {
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            // Skip header
            br.readLine();

            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                // Clean the values and trim any whitespace
                for (int i = 0; i < values.length; i++) {
                    values[i] = values[i].trim();
                }

                HousingData data = new HousingData(values);
                housingDataList.add(data);
            }

            System.out.println("Successfully loaded " + housingDataList.size() + " housing records.");

            // Normalize the data after loading
            normalizeData();

        } catch (IOException e) {
            System.err.println("Error loading data: " + e.getMessage());
        }
    }

    // Normalize the data for neural network training
    private void normalizeData() {
        if (housingDataList.isEmpty()) {
            return;
        }

        // Initialize min/max arrays based on the number of features
        int numFeatures = housingDataList.get(0).getRawFeatures().length;
        minFeatures = new double[numFeatures];
        maxFeatures = new double[numFeatures];

        // Initialize with first record
        double[] firstFeatures = housingDataList.get(0).getRawFeatures();
        System.arraycopy(firstFeatures, 0, minFeatures, 0, numFeatures);
        System.arraycopy(firstFeatures, 0, maxFeatures, 0, numFeatures);

        minPrice = housingDataList.get(0).getPrice();
        maxPrice = housingDataList.get(0).getPrice();

        // Find min and max for each feature and the price
        for (HousingData data : housingDataList) {
            double[] features = data.getRawFeatures();
            for (int i = 0; i < numFeatures; i++) {
                if (features[i] < minFeatures[i]) {
                    minFeatures[i] = features[i];
                }
                if (features[i] > maxFeatures[i]) {
                    maxFeatures[i] = features[i];
                }
            }

            if (data.getPrice() < minPrice) {
                minPrice = data.getPrice();
            }
            if (data.getPrice() > maxPrice) {
                maxPrice = data.getPrice();
            }
        }

        // Normalize each record
        for (HousingData data : housingDataList) {
            double[] features = data.getRawFeatures();
            double[] normalizedFeatures = new double[numFeatures];

            for (int i = 0; i < numFeatures; i++) {
                // Avoid division by zero
                if (maxFeatures[i] - minFeatures[i] == 0) {
                    normalizedFeatures[i] = 0.0;
                } else {
                    normalizedFeatures[i] = (features[i] - minFeatures[i]) / (maxFeatures[i] - minFeatures[i]);
                }
            }

            // Normalize price as the target
            double[] normalizedTarget = new double[1];
            normalizedTarget[0] = (data.getPrice() - minPrice) / (maxPrice - minPrice);

            data.setNormalizedFeatures(normalizedFeatures);
            data.setNormalizedTarget(normalizedTarget);
        }

        System.out.println("Data normalization completed.");
    }

    // Split data into training and testing sets
    public Map<String, List<HousingData>> splitData(double trainingRatio) {
        Collections.shuffle(housingDataList, new Random(42)); // Shuffle with fixed seed for reproducibility

        int trainingSize = (int) (housingDataList.size() * trainingRatio);
        List<HousingData> trainingData = new ArrayList<>(housingDataList.subList(0, trainingSize));
        List<HousingData> testingData = new ArrayList<>(housingDataList.subList(trainingSize, housingDataList.size()));

        Map<String, List<HousingData>> splitData = new HashMap<>();
        splitData.put("training", trainingData);
        splitData.put("testing", testingData);

        System.out.println("Data split: " + trainingData.size() + " training samples, " +
                testingData.size() + " testing samples");

        return splitData;
    }

    // Get feature dimensions for ANN setup
    public int getInputDimension() {
        if (housingDataList.isEmpty()) {
            return 0;
        }
        return housingDataList.get(0).getNormalizedFeatures().length;
    }

    // Get data in format ready for neural network
    public double[][] getFeatureMatrix(List<HousingData> dataList) {
        double[][] features = new double[dataList.size()][getInputDimension()];
        for (int i = 0; i < dataList.size(); i++) {
            features[i] = dataList.get(i).getNormalizedFeatures();
        }
        return features;
    }

    public double[][] getTargetMatrix(List<HousingData> dataList) {
        double[][] targets = new double[dataList.size()][1]; // 1 output (price)
        for (int i = 0; i < dataList.size(); i++) {
            targets[i] = dataList.get(i).getNormalizedTarget();
        }
        return targets;
    }

    // Denormalize the price prediction
    public double denormalizePrice(double normalizedPrice) {
        return normalizedPrice * (maxPrice - minPrice) + minPrice;
    }

    // Get all data
    public List<HousingData> getAllData() {
        return housingDataList;
    }

    public double[] getMinFeatures() {
        return minFeatures;
    }

    public double[] getMaxFeatures() {
        return maxFeatures;
    }

    public double getAvgArea() {
        if (housingDataList.isEmpty()) {
            return 0;
        }

        double totalArea = 0;
        for (HousingData data : housingDataList) {
            totalArea += data.getArea();
        }

        return totalArea / housingDataList.size();
    }

    


    public double[][] calculateCorrelationMatrix() {
        // Number of features + price
        int numFeatures = getInputDimension();
        int matrixSize = numFeatures + 1; // +1 for price
        double[][] correlationMatrix = new double[matrixSize][matrixSize];

        if (housingDataList.isEmpty()) {
            return correlationMatrix;
        }

        // Create a matrix of all data (features + price)
        double[][] allData = new double[housingDataList.size()][matrixSize];
        for (int i = 0; i < housingDataList.size(); i++) {
            HousingData data = housingDataList.get(i);
            double[] features = data.getRawFeatures();

            // Copy features
            System.arraycopy(features, 0, allData[i], 0, numFeatures);

            // Add price as the last column
            allData[i][matrixSize - 1] = data.getPrice();
        }

        // Calculate mean for each column
        double[] means = new double[matrixSize];
        for (int col = 0; col < matrixSize; col++) {
            double sum = 0;
            for (int row = 0; row < housingDataList.size(); row++) {
                sum += allData[row][col];
            }
            means[col] = sum / housingDataList.size();
        }

        // Calculate correlation matrix
        for (int i = 0; i < matrixSize; i++) {
            // Diagonal is always 1 (correlation of a variable with itself)
            correlationMatrix[i][i] = 1.0;

            for (int j = i + 1; j < matrixSize; j++) {
                // Calculate covariance
                double covariance = 0;
                double stdDevI = 0;
                double stdDevJ = 0;

                for (int k = 0; k < housingDataList.size(); k++) {
                    double diffI = allData[k][i] - means[i];
                    double diffJ = allData[k][j] - means[j];

                    covariance += diffI * diffJ;
                    stdDevI += diffI * diffI;
                    stdDevJ += diffJ * diffJ;
                }

                // Calculate correlation
                double correlation = 0;
                if (stdDevI > 0 && stdDevJ > 0) {
                    correlation = covariance / (Math.sqrt(stdDevI) * Math.sqrt(stdDevJ));
                }

                // Correlation is symmetric
                correlationMatrix[i][j] = correlation;
                correlationMatrix[j][i] = correlation;
            }
        }

        return correlationMatrix;
    }



    // Example main method to demonstrate usage
    public static void main(String[] args) {
        HousingDataLoader loader = new HousingDataLoader();

        // Load data from CSV
        loader.loadData("src/main/resources/Housing.csv");

        // Split into training and testing sets
        Map<String, List<HousingData>> splitData = loader.splitData(0.8); // 80% training, 20% testing

        // Get data in format ready for neural network
        List<HousingData> trainingData = splitData.get("training");
        double[][] trainingFeatures = loader.getFeatureMatrix(trainingData);
        double[][] trainingTargets = loader.getTargetMatrix(trainingData);

        List<HousingData> testingData = splitData.get("testing");
        double[][] testingFeatures = loader.getFeatureMatrix(testingData);
        double[][] testingTargets = loader.getTargetMatrix(testingData);

        // Print some statistics
        System.out.println("Training set size: " + trainingFeatures.length);
        System.out.println("Testing set size: " + testingFeatures.length);
        System.out.println("Feature dimension: " + loader.getInputDimension());

        // At this point, you could pass the training and testing data to your ANN model
        System.out.println("Data is ready for ANN model training");

        // Sample to show data format
        if (!trainingData.isEmpty()) {
            System.out.println("\nSample normalized features: ");
            for (int i = 0; i < Math.min(3, trainingData.size()); i++) {
                System.out.print("Sample " + i + " features: ");
                double[] features = trainingData.get(i).getNormalizedFeatures();
                for (double feature : features) {
                    System.out.printf("%.4f ", feature);
                }
                System.out.println();
                System.out.println("Target (normalized price): " + trainingData.get(i).getNormalizedTarget()[0]);
                System.out.println("Original price: " + trainingData.get(i).getPrice());
                System.out.println();
            }
        }
    }
}