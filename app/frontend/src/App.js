import React, { useState } from 'react';
import './App.css';

function App() {
  const [title, setTitle] = useState('');
  const [body, setBody] = useState('');
  const [alerts, setAlerts] = useState([]);
  const [riskLevel, setRiskLevel] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setAlerts([]);
    setRiskLevel('');
    setError('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, body }),
      });

      if (!response.ok) throw new Error('Error en el backend');
      const data = await response.json();

      setAlerts(data.alerts);
      setRiskLevel(data.risk_level);
    } catch (err) {
      setError('No se pudo conectar con el backend.');
    } finally {
      setLoading(false);
    }
  };

  const renderRiskLabel = (level) => {
    const map = {
      high: ['Alto riesgo', 'danger'],
      moderate: ['Riesgo moderado', 'warning'],
      low: ['Bajo riesgo', 'success'],
    };
    const [label, variant] = map[level] || ['', 'secondary'];
    return <span className={`badge bg-${variant} fs-6`}>{label}</span>;
  };

  const renderAlertClass = (text) => {
    if (text.startsWith('🔴')) return 'danger';
    if (text.startsWith('🟡')) return 'warning';
    if (text.startsWith('🟢')) return 'success';
    return 'secondary';
  };

  return (
    <div className="container mt-5">
      <h1 className="mb-4 text-center">📰 Verificador de Noticias</h1>

      <form onSubmit={handleSubmit} className="mb-4">
        <div className="mb-3">
          <input
            type="text"
            placeholder="Título de la noticia"
            className="form-control"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            required
          />
        </div>

        <div className="mb-3">
          <textarea
            placeholder="Texto de la noticia"
            className="form-control"
            value={body}
            onChange={(e) => setBody(e.target.value)}
            rows={10}
            required
          ></textarea>
        </div>

        <button className="btn btn-primary" type="submit" disabled={loading}>
          {loading ? 'Analizando...' : 'Analizar'}
        </button>
      </form>

      {error && <div className="alert alert-danger">{error}</div>}

      {alerts.length > 0 && (
        <div className="card shadow-sm">
          <div className="card-body">
            <h4 className="card-title">Resultado del análisis</h4>
            <div className="mb-3">{renderRiskLabel(riskLevel)}</div>

            <ul className="list-group">
              {alerts.map((alert, i) => (
                <li
                  key={i}
                  className={`list-group-item list-group-item-${renderAlertClass(alert)}`}
                >
                  {alert}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
