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
  
        if (!data || !data.csv) {
          throw new Error("Invalid response format");
        }
  
        // Extract the header and first row from the CSV
        const lines = data.csv.trim().split('\n');
        const headers = lines[0].split(',');
        const values = lines[1].split(',');
  
        const labels = headers.slice(1).map(header => header.replace('_ratio', ''));
        const yValues = values.slice(1).map(v => parseFloat(v));
  
        // Color palette
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
            text: yValues.map(val => val.toFixed(2)),
            textposition: 'outside', // place values above the bars
            textfont: {
              size: 16,             // increase font size
              color: '#fff',        // white or any contrasting color
              family: 'Segoe UI'    // optional: match your app font
            }
          };
  
        const layout = {
          title: `${mixtureFile.name} Result`,
          yaxis: { title: 'Concentration Ratio' },
          xaxis: { title: 'Ingredients' },
          margin: { t: 60 },
          plot_bgcolor: '#2c2f36',
          paper_bgcolor: '#2c2f36',
          font: { color: 'white' }
        };
  
        Plotly.newPlot('graph', [trace], layout);
        document.getElementById('graphOverlay').style.display = 'none';
        document.getElementById('csvOutput').textContent = data.csv;// Format CSV output
        const formatCsvOutput = (rawCsv, mixtureName) => {
          const lines = rawCsv.trim().split('\n');
          if (lines.length < 2) return "Invalid CSV data";
        
          const headers = lines[0].split(',').slice(1); // skip mixture column
          const values = lines[1].split(',').slice(1);
        
          let formatted = `Estimated Concentration Ratio of ${mixtureName}\n`;
          formatted += '----------------------------------------------------------------\n';

          for (let i = 0; i < headers.length; i++) {
            const name = headers[i].replace('_ratio', '');
            const val = values[i];
            formatted += `${name}: ${val}\n`;
          }
        
          formatted += '----------------------------------------------------------------';
          return formatted;
        };
        
        document.getElementById('csvOutput').textContent = formatCsvOutput(data.csv, mixtureFile.name);
        document.getElementById('loadingIndicator').style.display = 'none';

        
      } catch (error) {
        console.error('Analysis failed:', error);
        alert("Analysis failed. Check console for details.");
        document.getElementById('loadingIndicator').style.display = 'none';
      }
    });
  });
  