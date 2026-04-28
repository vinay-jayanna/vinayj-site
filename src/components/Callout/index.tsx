import type { ReactNode } from 'react';
import styles from './styles.module.css';

type CalloutType = 'insight' | 'warning' | 'formula' | 'tip' | 'definition';

interface CalloutProps {
  type: CalloutType;
  title?: string;
  children: ReactNode;
}

const CALLOUT_CONFIG: Record<CalloutType, { label: string; className: string }> = {
  insight: {
    label: 'Key Insight',
    className: styles.insight,
  },
  warning: {
    label: 'Watch Out',
    className: styles.warning,
  },
  formula: {
    label: 'Formula',
    className: styles.formula,
  },
  tip: {
    label: 'Production Tip',
    className: styles.tip,
  },
  definition: {
    label: 'Definition',
    className: styles.definition,
  },
};

export default function Callout({
  type,
  title,
  children,
}: CalloutProps): ReactNode {
  const config = CALLOUT_CONFIG[type];
  const label = title ?? config.label;

  return (
    <div className={`${styles.callout} ${config.className}`}>
      <div className={styles.calloutLabel}>{label}</div>
      <div className={styles.calloutBody}>{children}</div>
    </div>
  );
}
