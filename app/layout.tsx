import type { Metadata, Viewport } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'SamacharSetu · Understand the news',
  description: 'A multilingual news reader and hands-on NLP learning lab. English, Hindi and Marathi classifiers trained on real headlines.',
  openGraph: {
    title: 'SamacharSetu · Understand the news',
    description: 'Multilingual news reader with an integrated NLP learning lab.',
    type: 'website',
  },
};

// Viewport must be a separate export in Next.js 13+; cannot be nested inside metadata.
export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // lang="en" is the server-rendered default; the client-side interface-language
  // switcher (App.tsx) updates document.documentElement.lang reactively.
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
