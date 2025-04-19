document.addEventListener('DOMContentLoaded', () => {
  const runButton = document.getElementById('runButton');

  runButton.addEventListener('click', async () => {
    document.getElementById('loadingIndicator').style.display = 'flex';

    const mixtureInput = document.getElementById('mixtureInput');
    const ingredientInput = document.getElementById('ingredientInput');

    if (!mixtureInput || !ingredientInput || !mixtureInput.files.length || !ingredientInput.files.length) {
      alert("Please upload a mixture image and at least one ingredient image.");
      document.getElementById('loadingIndicator').style.display = 'none';
      return;
    }

    const formData = new FormData();
    const mixtureFile = mixtureInput.files[0];
    formData.append('mixtureImage', mixtureFile);

    for (const file of ingredientInput.files) {
      formData.append('ingredientImages', file);
    }

    const concentrationInput = document.getElementById('concentrationInput');
    const concentrationString = concentrationInput.value.trim();
    formData.append('concentrations', concentrationString);

    try {
      const response = await fetch("/run-analysis", {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!data || data.Solved === false || !data.Ratios || Object.keys(data.Ratios).length === 0) {
        document.getElementById('csvOutput').textContent = "⚠️ No solution could be found with the current input. Try adjusting concentrations or verify image quality.";
        Plotly.purge('graph');
        document.getElementById('loadingIndicator').style.display = 'none';
        return;
      }

      // Prepare graph data
      const labels = Object.keys(data.Ratios).map(key => key.replace('_ratio', ''));
      const yValues = Object.values(data.Ratios).map(val => val); // convert to percentage

      const barColors = [
        '#e74c3c', '#2ecc71', '#3498db', '#9b59b6', '#f39c12',
        '#1abc9c', '#e67e22', '#34495e', '#d35400', '#7f8c8d'
      ];

      const trace = {
        x: labels,
        y: yValues,
        type: 'bar',
        marker: {
          color: barColors.slice(0, yValues.length)
        },
        text: yValues.map(val => `${val.toFixed(2)}`),
        textposition: 'outside',
        textfont: {
          size: 16,
          color: '#fff',
          family: 'Segoe UI'
        }
      };

      const layout = {
        title: `${data.Mixture} Result`,
        yaxis: { title: 'Concentration Values' },
        xaxis: { title: 'Ingredients' },
        margin: { t: 60 },
        plot_bgcolor: '#2c2f36',
        paper_bgcolor: '#2c2f36',
        font: { color: 'white' }
      };

      Plotly.newPlot('graph', [trace], layout);
      document.getElementById('graphOverlay').style.display = 'none';

      // Pretty text output
      const formatJsonOutput = (data) => {
        const lines = [];
        lines.push(`Estimated Concentration Ratio of ${data.Mixture}`);
        lines.push('----------------------------------------------------------------');
        for (const [key, value] of Object.entries(data.Ratios)) {
          const name = key.replace('_ratio', '');
          lines.push(`${name}: ${(value)}`);
        }
        lines.push('----------------------------------------------------------------');
        return lines.join('\n');
      };

      document.getElementById('csvOutput').textContent = formatJsonOutput(data);
      document.getElementById('loadingIndicator').style.display = 'none';

    } catch (error) {
      console.error('Analysis failed:', error);
      alert("Analysis failed. Check console for details.");
      document.getElementById('loadingIndicator').style.display = 'none';
    }
  });
});
