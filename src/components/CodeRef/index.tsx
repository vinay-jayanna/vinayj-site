import type { ReactNode } from 'react';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

interface CodeRefProps {
  repo: string;
  file: string;
  description: string;
  language?: string;
  href: string;
}

export default function CodeRef({
  repo,
  file,
  description,
  language,
  href,
}: CodeRefProps): ReactNode {
  return (
    <Link to={href} className={styles.codeRef}>
      <div className={styles.left}>
        <div className={styles.icon}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round"
            strokeLinejoin="round">
            <polyline points="16 18 22 12 16 6" />
            <polyline points="8 6 2 12 8 18" />
          </svg>
        </div>
        <div className={styles.meta}>
          <div className={styles.fileRow}>
            <span className={styles.repo}>{repo}</span>
            <span className={styles.sep}>/</span>
            <span className={styles.file}>{file}</span>
            {language && (
              <span className={styles.lang}>{language}</span>
            )}
          </div>
          <div className={styles.description}>{description}</div>
        </div>
      </div>
      <div className={styles.arrow}>→</div>
    </Link>
  );
}
