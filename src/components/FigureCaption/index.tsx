import type { ReactNode } from 'react';
import styles from './styles.module.css';

interface FigureCaptionProps {
  src: string;
  alt: string;
  caption: string;
  number?: string;
  wide?: boolean;
}

export default function FigureCaption({
  src,
  alt,
  caption,
  number,
  wide = false,
}: FigureCaptionProps): ReactNode {
  return (
    <figure className={`${styles.figure} ${wide ? styles.wide : ''}`}>
      <img src={src} alt={alt} className={styles.image} />
      <figcaption className={styles.caption}>
        {number && <span className={styles.figureNumber}>Figure {number} — </span>}
        {caption}
      </figcaption>
    </figure>
  );
}
