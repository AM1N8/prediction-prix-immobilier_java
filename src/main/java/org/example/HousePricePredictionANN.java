package org.example;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableColumn;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class HousePricePredictionANN {
    private MultiLayerNetwork model;
    private HousingDataLoader dataLoader;
    private JFrame frame;
    private JTextField areaField, bedroomsField, bathroomsField, storiesField, parkingField;
    private JComboBox<String> furnishingStatusBox;
    private JCheckBox mainroadCheck, guestroomCheck, basementCheck, hotwaterCheck, acCheck, prefareaCheck;
    private JTextArea resultArea;
    private JTabbedPane tabbedPane;
    private JPanel correlationPanel;
    private JPanel modelDescriptionPanel;

    // Fixed path to the housing dataset (to be packaged with the application)
    private static final String DEFAULT_DATASET_PATH = "src/main/resources/Housing.csv";

    // Conversion rate from INR to USD (as of May 2025)
    private static final double INR_TO_USD_RATE = 0.012;

    // Feature names for correlation matrix
    private static final String[] FEATURE_NAMES = {
            "area", "bedrooms", "bathrooms", "stories", "mainroad",
            "guestroom", "basement", "hotwater", "airconditioning",
            "parking", "prefarea", "furnishing_status", "price"
    };

    public HousePricePredictionANN() {
        dataLoader = new HousingDataLoader();
        // Load data immediately using the default path
        dataLoader.loadData(DEFAULT_DATASET_PATH);
    }

    public void buildModel() {
        // Get input dimension from the data loader
        int numInputs = dataLoader.getInputDimension();
        int numOutputs = 1; // Price prediction
        int numHiddenNodes = 20;

        // Neural network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(numHiddenNodes)
                        .nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(numHiddenNodes)
                        .nOut(numOutputs)
                        .activation(Activation.IDENTITY)
                        .build())
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));
    }

    public void trainModel() {
        // Split data
        Map<String, List<HousingDataLoader.HousingData>> splitData = dataLoader.splitData(0.8);
        List<HousingDataLoader.HousingData> trainingData = splitData.get("training");

        // Convert to ND4J format
        double[][] trainingFeatures = dataLoader.getFeatureMatrix(trainingData);
        double[][] trainingTargets = dataLoader.getTargetMatrix(trainingData);

        INDArray featuresNDArray = Nd4j.create(trainingFeatures);
        INDArray targetsNDArray = Nd4j.create(trainingTargets);

        DataSet trainingSet = new DataSet(featuresNDArray, targetsNDArray);

        // Train the model
        for (int i = 0; i < 1000; i++) {
            model.fit(trainingSet);
            if (i % 100 == 0) {
                System.out.println("Epoch " + i + ", Score: " + model.score());
            }
        }
    }

    public Map<String, Double> evaluateModel() {
        Map<String, List<HousingDataLoader.HousingData>> splitData = dataLoader.splitData(0.8);
        List<HousingDataLoader.HousingData> testingData = splitData.get("testing");

        double[][] testingFeatures = dataLoader.getFeatureMatrix(testingData);
        double[][] testingTargets = dataLoader.getTargetMatrix(testingData);

        INDArray featuresNDArray = Nd4j.create(testingFeatures);
        INDArray targetsNDArray = Nd4j.create(testingTargets);

        INDArray predictions = model.output(featuresNDArray);

        // Calculate Mean Squared Error (MSE)
        INDArray diff = predictions.sub(targetsNDArray);
        INDArray squaredDiff = diff.mul(diff);
        double mse = squaredDiff.sumNumber().doubleValue() / predictions.rows();

        // Calculate RMSE (Root Mean Squared Error)
        double rmse = Math.sqrt(mse);

        // Calculate R² score
        double ssTot = 0.0;
        double mean = targetsNDArray.meanNumber().doubleValue();
        for (int i = 0; i < targetsNDArray.rows(); i++) {
            double diff_from_mean = targetsNDArray.getDouble(i, 0) - mean;
            ssTot += diff_from_mean * diff_from_mean;
        }

        double ssRes = squaredDiff.sumNumber().doubleValue();
        double r2 = 1 - (ssRes / ssTot);

        // Calculate original price predictions and targets for display
        DecimalFormat df = new DecimalFormat("#,###.##");
        System.out.println("\nTest Predictions:");
        double totalError = 0.0;
        int samplesToShow = Math.min(5, testingData.size());
        for (int i = 0; i < samplesToShow; i++) {
            double normalizedPrediction = predictions.getDouble(i, 0);
            double originalPrediction = dataLoader.denormalizePrice(normalizedPrediction);
            double originalPredictionUSD = originalPrediction * INR_TO_USD_RATE;
            double originalTarget = testingData.get(i).getPrice();
            double originalTargetUSD = originalTarget * INR_TO_USD_RATE;
            double error = Math.abs((originalPrediction - originalTarget) / originalTarget) * 100;
            totalError += error;

            System.out.println("Sample " + i + ": Predicted: " + df.format(originalPrediction) + " INR ($" +
                    df.format(originalPredictionUSD) + ")" +
                    ", Actual: " + df.format(originalTarget) + " INR ($" +
                    df.format(originalTargetUSD) + ")" +
                    ", Error: " + String.format("%.2f%%", error));
        }

        System.out.println("Average Error on Samples: " + String.format("%.2f%%", totalError / samplesToShow));
        System.out.println("MSE (normalized): " + mse);
        System.out.println("RMSE (normalized): " + rmse);
        System.out.println("R² Score: " + r2);

        // Return the metrics in a map
        Map<String, Double> metrics = new HashMap<>();
        metrics.put("mse", mse);
        metrics.put("rmse", rmse);
        metrics.put("r2", r2);
        metrics.put("averagePercentError", totalError / samplesToShow);

        // Update the correlation matrix panel
        updateCorrelationMatrix();

        return metrics;
    }

    private void updateCorrelationMatrix() {
        SwingUtilities.invokeLater(() -> {
            // Get correlation matrix from data loader
            double[][] correlationMatrix = dataLoader.calculateCorrelationMatrix();

            // Create a table model for the correlation matrix
            DefaultTableModel model = new DefaultTableModel();

            // Add column names
            for (String feature : FEATURE_NAMES) {
                model.addColumn(feature);
            }

            // Add rows with data
            DecimalFormat df = new DecimalFormat("0.00");
            for (int i = 0; i < correlationMatrix.length; i++) {
                Object[] row = new Object[correlationMatrix[i].length];
                for (int j = 0; j < correlationMatrix[i].length; j++) {
                    row[j] = df.format(correlationMatrix[i][j]);
                }
                model.addRow(row);
            }

            // Create table with the model
            JTable table = new JTable(model);
            table.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);

            // Calculate preferred width for each column based on content
            for (int column = 0; column < table.getColumnCount(); column++) {
                TableColumn tableColumn = table.getColumnModel().getColumn(column);
                int preferredWidth = Math.max(100, tableColumn.getPreferredWidth() + 10);
                tableColumn.setPreferredWidth(preferredWidth);
            }

            // Create row header table with proper sizing
            DefaultTableModel rowHeaderModel = new DefaultTableModel(0, 1);
            // Add feature names as rows
            for (String featureName : FEATURE_NAMES) {
                rowHeaderModel.addRow(new Object[]{featureName});
            }

            JTable rowHeader = new JTable(rowHeaderModel);
            rowHeader.setEnabled(false);

            // Set preferred width for row header based on content
            int maxRowHeaderWidth = 0;
            for (String feature : FEATURE_NAMES) {
                maxRowHeaderWidth = Math.max(maxRowHeaderWidth,
                        rowHeader.getFontMetrics(rowHeader.getFont())
                                .stringWidth(feature) + 20);
            }
            rowHeader.getColumnModel().getColumn(0).setPreferredWidth(maxRowHeaderWidth);
            rowHeader.setPreferredScrollableViewportSize(
                    new Dimension(maxRowHeaderWidth, rowHeader.getPreferredSize().height));

            // Apply a cell renderer to color cells based on correlation strength and ensure proper alignment
            table.setDefaultRenderer(Object.class, new DefaultTableCellRenderer() {
                @Override
                public Component getTableCellRendererComponent(JTable table, Object value,
                                                               boolean isSelected, boolean hasFocus, int row, int column) {
                    Component c = super.getTableCellRendererComponent(
                            table, value, isSelected, hasFocus, row, column);

                    try {
                        double val = Double.parseDouble(value.toString());
                        // Color based on correlation strength
                        if (val > 0.7 || val < -0.7) {
                            c.setBackground(new Color(255, 200, 200)); // Strong correlation
                        } else if (val > 0.4 || val < -0.4) {
                            c.setBackground(new Color(255, 230, 230)); // Moderate correlation
                        } else {
                            c.setBackground(Color.WHITE); // Weak correlation
                        }
                    } catch (NumberFormatException e) {
                        c.setBackground(Color.WHITE);
                    }

                    setHorizontalAlignment(SwingConstants.CENTER);
                    setBorder(BorderFactory.createEmptyBorder(2, 5, 2, 5));
                    return c;
                }
            });

            // Apply similar styling to row header
            rowHeader.setDefaultRenderer(Object.class, new DefaultTableCellRenderer() {
                @Override
                public Component getTableCellRendererComponent(JTable table, Object value,
                                                               boolean isSelected, boolean hasFocus, int row, int column) {
                    Component c = super.getTableCellRendererComponent(
                            table, value, isSelected, hasFocus, row, column);
                    setBackground(new Color(240, 240, 240));
                    setHorizontalAlignment(SwingConstants.LEFT);
                    setBorder(BorderFactory.createEmptyBorder(2, 5, 2, 5));
                    setFont(getFont().deriveFont(Font.BOLD));
                    return c;
                }
            });

            // Ensure both tables have the same row height
            rowHeader.setRowHeight(table.getRowHeight());

            // Clear old content and add the new table
            correlationPanel.removeAll();
            correlationPanel.setLayout(new BorderLayout());

            // Create header panel with logo and title
            JPanel headerPanel = new JPanel(new BorderLayout());

            // Load and add school logo
            JLabel logoLabel = new JLabel();
            try {
                // Load the ENSAM logo
                ImageIcon logoIcon = new ImageIcon(getClass().getResource("/LOGO_ENSAM.png"));

                // Calculate the scaling ratio to fit the height while maintaining aspect ratio
                Image img = logoIcon.getImage();
                double scaleFactor = 100.0 / img.getHeight(null); // Target height of 100px
                int scaledWidth = (int)(img.getWidth(null) * scaleFactor);

                // Scale the image proportionally
                Image scaledImg = img.getScaledInstance(scaledWidth, 100, Image.SCALE_SMOOTH);
                logoIcon = new ImageIcon(scaledImg);

                logoLabel.setIcon(logoIcon);
                logoLabel.setHorizontalAlignment(SwingConstants.CENTER);
                logoLabel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
                logoLabel.setPreferredSize(new Dimension(scaledWidth, 100));
            } catch (Exception e) {
                // Fallback if image loading fails
                logoLabel.setText("Logo ENSAM");
                logoLabel.setHorizontalAlignment(SwingConstants.CENTER);
                logoLabel.setBorder(BorderFactory.createLineBorder(Color.LIGHT_GRAY));
                logoLabel.setPreferredSize(new Dimension(240, 100));
                System.err.println("Failed to load logo image: " + e.getMessage());
            }

            // Add title with improved styling
            JLabel titleLabel = new JLabel("Matrice de Corrélation", SwingConstants.CENTER);
            titleLabel.setFont(new Font("Sans-Serif", Font.BOLD, 18));
            titleLabel.setBorder(BorderFactory.createEmptyBorder(0, 20, 0, 0));

            // Add logo and title to header
            JPanel logoPanel = new JPanel(new BorderLayout());
            logoPanel.add(logoLabel, BorderLayout.CENTER);
            logoPanel.setBorder(BorderFactory.createEmptyBorder(5, 10, 5, 20));

            headerPanel.add(logoPanel, BorderLayout.WEST);
            headerPanel.add(titleLabel, BorderLayout.CENTER);

            // Add padding around the header
            headerPanel.setBorder(BorderFactory.createCompoundBorder(
                    BorderFactory.createMatteBorder(0, 0, 1, 0, Color.LIGHT_GRAY),
                    BorderFactory.createEmptyBorder(10, 10, 10, 10)
            ));

            // Create a scroll pane with the table
            JScrollPane scrollPane = new JScrollPane(table);
            scrollPane.setRowHeaderView(rowHeader);

            // Make sure horizontal scrollbar always shows
            scrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);

            // Set preferred size for the scroll pane
            scrollPane.setPreferredSize(new Dimension(650, 400));

            // Change the layout structure
            correlationPanel.add(headerPanel, BorderLayout.NORTH);
            correlationPanel.add(scrollPane, BorderLayout.CENTER);

            // Add legend in a more structured way
            JPanel legendPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
            legendPanel.setBorder(BorderFactory.createTitledBorder("Légende"));
            addLegendItem(legendPanel, "Forte corrélation (>0.7)", new Color(255, 200, 200));
            addLegendItem(legendPanel, "Corrélation modérée (>0.4)", new Color(255, 230, 230));
            addLegendItem(legendPanel, "Faible corrélation (<0.4)", Color.WHITE);
            correlationPanel.add(legendPanel, BorderLayout.SOUTH);

            correlationPanel.revalidate();
            correlationPanel.repaint();
        });
    }

    private void addLegendItem(JPanel panel, String text, Color color) {
        JPanel colorBox = new JPanel();
        colorBox.setBackground(color);
        colorBox.setPreferredSize(new Dimension(20, 20));
        colorBox.setBorder(BorderFactory.createLineBorder(Color.BLACK));
        panel.add(colorBox);
        panel.add(new JLabel(text));
        panel.add(Box.createHorizontalStrut(15));
    }

    public double predictPrice(double[] features) {
        // Make sure features are normalized the same way as training data
        int numFeatures = dataLoader.getInputDimension();
        double[] normalizedFeatures = new double[numFeatures];

        // Get min/max values
        double[] minFeatures = dataLoader.getMinFeatures();
        double[] maxFeatures = dataLoader.getMaxFeatures();

        for (int i = 0; i < numFeatures; i++) {
            // Avoid division by zero
            if (maxFeatures[i] - minFeatures[i] == 0) {
                normalizedFeatures[i] = 0.0;
            } else {
                normalizedFeatures[i] = (features[i] - minFeatures[i]) / (maxFeatures[i] - minFeatures[i]);
            }
        }

        // Create ND4j array with batch size 1
        INDArray input = Nd4j.create(new double[][]{normalizedFeatures});

        // Get model prediction (normalized)
        INDArray output = model.output(input);
        double normalizedPrediction = output.getDouble(0, 0);

        // Convert back to original price scale
        return dataLoader.denormalizePrice(normalizedPrediction);
    }

    public void createAndShowGUI() {
        // Create the main frame
        frame = new JFrame("Prédiction de Prix de Maison");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 700);

        // Create tabbed pane
        tabbedPane = new JTabbedPane();

        // Create panels for each tab
        JPanel inputPanel = createInputPanel();
        modelDescriptionPanel = createModelDescriptionPanel();
        correlationPanel = new JPanel();

        // Add tabs
        tabbedPane.addTab("Prédiction", inputPanel);
        tabbedPane.addTab("Description du Modèle", modelDescriptionPanel);
        tabbedPane.addTab("Corrélations", correlationPanel);

        // Add tabbed pane to frame
        frame.add(tabbedPane);

        // Display the frame
        frame.setVisible(true);
    }

    private JPanel createInputPanel() {
        JPanel panel = new JPanel(new BorderLayout(10, 10));
        panel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));

        // Input fields panel
        JPanel controlPanel = new JPanel();
        controlPanel.setLayout(new GridLayout(0, 2, 10, 10));

        // Input fields
        areaField = new JTextField(10);
        bedroomsField = new JTextField(10);
        bathroomsField = new JTextField(10);
        storiesField = new JTextField(10);
        parkingField = new JTextField(10);

        // Checkboxes
        mainroadCheck = new JCheckBox("Oui");
        guestroomCheck = new JCheckBox("Oui");
        basementCheck = new JCheckBox("Oui");
        hotwaterCheck = new JCheckBox("Oui");
        acCheck = new JCheckBox("Oui");
        prefareaCheck = new JCheckBox("Oui");

        // Dropdown for furnishing status
        String[] furnishingOptions = {"meublé", "semi-meublé", "non meublé"};
        furnishingStatusBox = new JComboBox<>(furnishingOptions);

        // Add components to control panel
        controlPanel.add(new JLabel("Surface (m²):"));
        controlPanel.add(areaField);
        controlPanel.add(new JLabel("Chambres:"));
        controlPanel.add(bedroomsField);
        controlPanel.add(new JLabel("Salles de bain:"));
        controlPanel.add(bathroomsField);
        controlPanel.add(new JLabel("Étages:"));
        controlPanel.add(storiesField);
        controlPanel.add(new JLabel("Places de parking:"));
        controlPanel.add(parkingField);
        controlPanel.add(new JLabel("Route principale:"));
        controlPanel.add(mainroadCheck);
        controlPanel.add(new JLabel("Chambre d'amis:"));
        controlPanel.add(guestroomCheck);
        controlPanel.add(new JLabel("Sous-sol:"));
        controlPanel.add(basementCheck);
        controlPanel.add(new JLabel("Chauffage eau chaude:"));
        controlPanel.add(hotwaterCheck);
        controlPanel.add(new JLabel("Climatisation:"));
        controlPanel.add(acCheck);
        controlPanel.add(new JLabel("Zone préférentielle:"));
        controlPanel.add(prefareaCheck);
        controlPanel.add(new JLabel("État d'ameublement:"));
        controlPanel.add(furnishingStatusBox);

        // Create button panel
        JPanel buttonPanel = new JPanel();
        JButton predictButton = new JButton("Prédire le Prix");
        JButton trainButton = new JButton("Entraîner le Modèle");
        buttonPanel.add(trainButton);
        buttonPanel.add(predictButton);

        // Create results area
        resultArea = new JTextArea(10, 40);
        resultArea.setEditable(false);
        JScrollPane scrollPane = new JScrollPane(resultArea);

        // Add components to panel
        panel.add(controlPanel, BorderLayout.NORTH);
        panel.add(buttonPanel, BorderLayout.CENTER);
        panel.add(scrollPane, BorderLayout.SOUTH);

        // Predict button action
        predictButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    // Get input values from GUI
                    double area = Double.parseDouble(areaField.getText());
                    int bedrooms = Integer.parseInt(bedroomsField.getText());
                    int bathrooms = Integer.parseInt(bathroomsField.getText());
                    int stories = Integer.parseInt(storiesField.getText());
                    int parking = Integer.parseInt(parkingField.getText());
                    boolean mainroad = mainroadCheck.isSelected();
                    boolean guestroom = guestroomCheck.isSelected();
                    boolean basement = basementCheck.isSelected();
                    boolean hotwater = hotwaterCheck.isSelected();
                    boolean ac = acCheck.isSelected();
                    boolean prefarea = prefareaCheck.isSelected();

                    // Get furnishing status
                    String furnishingStatus;
                    switch (furnishingStatusBox.getSelectedIndex()) {
                        case 0:
                            furnishingStatus = "furnished";
                            break;
                        case 1:
                            furnishingStatus = "semi-furnished";
                            break;
                        default:
                            furnishingStatus = "unfurnished";
                            break;
                    }

                    // Prepare features array in the same order as in HousingData.getRawFeatures()
                    double[] features = new double[12];
                    features[0] = area;
                    features[1] = bedrooms;
                    features[2] = bathrooms;
                    features[3] = stories;
                    features[4] = mainroad ? 1.0 : 0.0;
                    features[5] = guestroom ? 1.0 : 0.0;
                    features[6] = basement ? 1.0 : 0.0;
                    features[7] = hotwater ? 1.0 : 0.0;
                    features[8] = ac ? 1.0 : 0.0;
                    features[9] = parking;
                    features[10] = prefarea ? 1.0 : 0.0;

                    // Furnishing status one-hot encoding
                    if (furnishingStatus.equals("furnished")) {
                        features[11] = 0.0;
                    } else if (furnishingStatus.equals("semi-furnished")) {
                        features[11] = 1.0;
                    } else { // unfurnished
                        features[11] = 2.0;
                    }

                    // Predict price
                    double predictedPrice = predictPrice(features);
                    double predictedPriceUSD = predictedPrice * INR_TO_USD_RATE;
                    DecimalFormat df = new DecimalFormat("#,###.##");
                    resultArea.setText("Prix prédit: " + df.format(predictedPrice) + " INR\n");
                    resultArea.append("Prix prédit (USD): $" + df.format(predictedPriceUSD) + "\n");

                    // Display feature importance if model is trained
                    if (model != null) {
                        resultArea.append("\nFacteurs les plus influents:\n");
                        // This is a simplistic approach - in a real app we'd extract this from the model
                        if (area > dataLoader.getAvgArea()) {
                            resultArea.append("- Grande surface (+)\n");
                        }
                        if (prefarea) {
                            resultArea.append("- Zone préférentielle (+)\n");
                        }
                        if (ac) {
                            resultArea.append("- Présence de climatisation (+)\n");
                        }
                    }

                } catch (NumberFormatException ex) {
                    resultArea.setText("Erreur: Veuillez entrer des valeurs numériques valides.");
                } catch (Exception ex) {
                    resultArea.setText("Erreur: " + ex.getMessage());
                }
            }
        });

        // Train button action
        trainButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                SwingWorker<Void, String> worker = new SwingWorker<Void, String>() {
                    @Override
                    protected Void doInBackground() {
                        publish("Début de l'entraînement du modèle...");
                        try {
                            publish("Construction du modèle...");
                            buildModel();

                            publish("Entraînement du modèle...");
                            trainModel();

                            publish("Évaluation du modèle...");
                            Map<String, Double> metrics = evaluateModel();

                            // Format the metrics for display
                            publish("Entraînement terminé!");
                            publish("MSE (normalisé): " + String.format("%.5f", metrics.get("mse")));
                            publish("RMSE (normalisé): " + String.format("%.5f", metrics.get("rmse")));
                            publish("R² Score: " + String.format("%.5f", metrics.get("r2")));
                            publish("Erreur moyenne: " + String.format("%.2f%%", metrics.get("averagePercentError")));

                            // Display sample predictions with USD conversion
                            Map<String, List<HousingDataLoader.HousingData>> splitData = dataLoader.splitData(0.8);
                            List<HousingDataLoader.HousingData> testingData = splitData.get("testing");
                            DecimalFormat df = new DecimalFormat("#,###.##");

                            int samplesToShow = Math.min(3, testingData.size());
                            publish("\nExemples de prédictions:");
                            for (int i = 0; i < samplesToShow; i++) {
                                double[] features = testingData.get(i).getRawFeatures();
                                double actualPrice = testingData.get(i).getPrice();
                                double actualPriceUSD = actualPrice * INR_TO_USD_RATE;
                                double predictedPrice = predictPrice(features);
                                double predictedPriceUSD = predictedPrice * INR_TO_USD_RATE;

                                publish("Exemple " + (i+1) + ":");
                                publish("  Réel: " + df.format(actualPrice) + " INR ($" + df.format(actualPriceUSD) + ")");
                                publish("  Prédit: " + df.format(predictedPrice) + " INR ($" + df.format(predictedPriceUSD) + ")");
                            }

                            publish("\nLe modèle est prêt pour les prédictions.");

                            // Update the model description panel
                            updateModelDescription(metrics);

                        } catch (Exception ex) {
                            publish("Erreur: " + ex.getMessage());
                            ex.printStackTrace();
                        }
                        return null;
                    }

                    @Override
                    protected void process(List<String> chunks) {
                        for (String message : chunks) {
                            resultArea.append(message + "\n");
                        }
                    }
                };
                worker.execute();
            }
        });

        return panel;
    }

    private JPanel createModelDescriptionPanel() {
        JPanel panel = new JPanel(new BorderLayout(10, 10));
        panel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));

        // Title
        JLabel titleLabel = new JLabel("Description du Modèle de Réseau de Neurones", SwingConstants.CENTER);
        titleLabel.setFont(new Font("Sans-Serif", Font.BOLD, 16));
        panel.add(titleLabel, BorderLayout.NORTH);

        // Model description
        JTextArea descriptionArea = new JTextArea();
        descriptionArea.setEditable(false);
        descriptionArea.setLineWrap(true);
        descriptionArea.setWrapStyleWord(true);
        descriptionArea.setFont(new Font("Sans-Serif", Font.PLAIN, 14));

        // Set initial description text
        descriptionArea.setText(
                "Architecture du Modèle:\n\n" +
                        "Ce système utilise un réseau de neurones artificiels (RNA) pour prédire les prix immobiliers. " +
                        "Le modèle est structuré comme suit:\n\n" +
                        "• Couche d'entrée: 12 neurones (correspondant aux caractéristiques du logement)\n" +
                        "• Première couche cachée: 20 neurones avec activation ReLU\n" +
                        "• Deuxième couche cachée: 20 neurones avec activation ReLU\n" +
                        "• Couche de sortie: 1 neurone (prédiction du prix)\n\n" +
                        "Algorithme d'optimisation: Adam (Adaptive Moment Estimation)\n" +
                        "Fonction de perte: Erreur quadratique moyenne (MSE)\n" +
                        "Initialisation des poids: Xavier\n\n" +
                        "Caractéristiques utilisées:\n" +
                        "• Surface (en m²)\n" +
                        "• Nombre de chambres\n" +
                        "• Nombre de salles de bain\n" +
                        "• Nombre d'étages\n" +
                        "• Proximité d'une route principale (oui/non)\n" +
                        "• Présence d'une chambre d'amis (oui/non)\n" +
                        "• Présence d'un sous-sol (oui/non)\n" +
                        "• Système d'eau chaude (oui/non)\n" +
                        "• Climatisation (oui/non)\n" +
                        "• Nombre de places de parking\n" +
                        "• Zone préférentielle (oui/non)\n" +
                        "• État d'ameublement (meublé/semi-meublé/non meublé)\n\n" +
                        "Performances du modèle: (apparaîtront après l'entraînement)"
        );

        JScrollPane scrollPane = new JScrollPane(descriptionArea);
        panel.add(scrollPane, BorderLayout.CENTER);

        // Add a visualization panel for the neural network architecture
        JPanel nnVisualPanel = createNeuralNetworkVisualization();
        panel.add(nnVisualPanel, BorderLayout.SOUTH);

        return panel;
    }

    private JPanel createNeuralNetworkVisualization() {
        JPanel panel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2d = (Graphics2D) g;

                // Enable anti-aliasing
                g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

                // Draw neural network structure
                int width = getWidth();
                int height = getHeight();

                // Define the layers
                int[] layers = {12, 20, 20, 1};
                int maxNeurons = 20;

                // Calculate spacing
                int layerSpacing = width / (layers.length + 1);
                int maxNeuronHeight = height / (maxNeurons + 1);

                // Colors
                Color inputColor = new Color(135, 206, 250);    // Light blue
                Color hiddenColor = new Color(255, 165, 0);     // Orange
                Color outputColor = new Color(152, 251, 152);   // Light green
                Color lineColor = new Color(200, 200, 200);     // Light gray

                // Draw connections first so they appear behind neurons
                g2d.setStroke(new BasicStroke(0.5f));
                g2d.setColor(lineColor);

                // Only draw some representative connections to avoid visual clutter
                for (int layer = 0; layer < layers.length - 1; layer++) {
                    int x1 = (layer + 1) * layerSpacing;
                    int x2 = (layer + 2) * layerSpacing;

                    int numNeurons1 = Math.min(layers[layer], 20); // Limit visible neurons
                    int numNeurons2 = Math.min(layers[layer + 1], 20);

                    // Calculate vertical spacing for neurons
                    int spacing1 = height / (numNeurons1 + 1);
                    int spacing2 = height / (numNeurons2 + 1);

                    // Draw sample connections (not all to avoid clutter)
                    int connectionStep1 = Math.max(1, numNeurons1 / 5);
                    int connectionStep2 = Math.max(1, numNeurons2 / 5);

                    for (int i = 0; i < numNeurons1; i += connectionStep1) {
                        int y1 = (i + 1) * spacing1;

                        for (int j = 0; j < numNeurons2; j += connectionStep2) {
                            int y2 = (j + 1) * spacing2;
                            g2d.drawLine(x1, y1, x2, y2);
                        }
                    }
                }

                // Draw neurons
                for (int layer = 0; layer < layers.length; layer++) {
                    int x = (layer + 1) * layerSpacing;
                    int numNeurons = Math.min(layers[layer], 20); // Limit visible neurons
                    int neuronSpacing = height / (numNeurons + 1);

                    // Select color based on layer
                    if (layer == 0) {
                        g2d.setColor(inputColor);
                    } else if (layer == layers.length - 1) {
                        g2d.setColor(outputColor);
                    } else {
                        g2d.setColor(hiddenColor);
                    }

                    // Draw neurons
                    for (int i = 0; i < numNeurons; i++) {
                        int y = (i + 1) * neuronSpacing;
                        g2d.fillOval(x - 10, y - 10, 20, 20);
                    }

                    // If there are more neurons than we're showing, draw ellipses to indicate more
                    if (layers[layer] > 20) {
                        g2d.fillOval(x - 2, height - 40, 4, 4);
                        g2d.fillOval(x - 2, height - 30, 4, 4);
                        g2d.fillOval(x - 2, height - 20, 4, 4);
                    }
                }

                // Draw layer labels
                g2d.setColor(Color.BLACK);
                g2d.setFont(new Font("Sans-Serif", Font.BOLD, 12));
                g2d.drawString("Entrée", layerSpacing - 20, height - 5);
                g2d.drawString("Cachée 1", 2 * layerSpacing - 30, height - 5);
                g2d.drawString("Cachée 2", 3 * layerSpacing - 30, height - 5);
                g2d.drawString("Sortie", 4 * layerSpacing - 20, height - 5);
            }
        };

        panel.setPreferredSize(new Dimension(600, 300));
        panel.setBorder(BorderFactory.createTitledBorder("Architecture du Réseau de Neurones"));

        return panel;
    }

    private void updateModelDescription(Map<String, Double> metrics) {
        SwingUtilities.invokeLater(() -> {
            // Find the text area in the description panel
            for (Component comp : modelDescriptionPanel.getComponents()) {
                if (comp instanceof JScrollPane) {
                    JViewport viewport = ((JScrollPane) comp).getViewport();
                    if (viewport.getView() instanceof JTextArea) {
                        JTextArea textArea = (JTextArea) viewport.getView();

                        // Update the text with metrics
                        String baseText = textArea.getText();
                        int performanceIndex = baseText.indexOf("Performances du modèle:");
                        if (performanceIndex != -1) {
                            String newText = baseText.substring(0, performanceIndex) +
                                    "Performances du modèle:\n\n" +
                                    "• MSE (normalisé): " + String.format("%.5f", metrics.get("mse")) + "\n" +
                                    "• RMSE (normalisé): " + String.format("%.5f", metrics.get("rmse")) + "\n" +
                                    "• R² Score: " + String.format("%.5f", metrics.get("r2")) + "\n" +
                                    "• Erreur moyenne: " + String.format("%.2f%%", metrics.get("averagePercentError")) + "\n\n" +
                                    "Le modèle a été entraîné sur 80% des données et testé sur les 20% restants.\n" +
                                    "Le score R² indique que le modèle explique environ " +
                                    String.format("%.1f%%", metrics.get("r2") * 100) +
                                    " de la variance des prix immobiliers.";

                            textArea.setText(newText);
                        }
                        break;
                    }
                }
            }
        });
    }

    public static void main(String[] args) {
        // Set up and show GUI
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                try {
                    // Set Look and Feel to system style
                    UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
                } catch (Exception e) {
                    e.printStackTrace();
                }

                HousePricePredictionANN app = new HousePricePredictionANN();
                app.createAndShowGUI();
            }
        });
    }
}