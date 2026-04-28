import type { ReactNode } from 'react';
import styles from './styles.module.css';

interface FormulaBlockProps {
  formula: string;
  label?: string;
  explanation?: string;
  variables?: { symbol: string; meaning: string }[];
}

export default function FormulaBlock({
  formula,
  label,
  explanation,
  variables,
}: FormulaBlockProps): ReactNode {
  return (
    <div className={styles.block}>
      {label && <div className={styles.label}>{label}</div>}
      <div className={styles.formula}>{formula}</div>
      {explanation && <p className={styles.explanation}>{explanation}</p>}
      {variables && variables.length > 0 && (
        <div className={styles.variables}>
          {variables.map(({ symbol, meaning }) => (
            <div key={symbol} className={styles.variableRow}>
              <code className={styles.symbol}>{symbol}</code>
              <span className={styles.meaning}>{meaning}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
