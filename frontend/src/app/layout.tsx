import type { Metadata } from "next";
import "./globals.css";
import Link from "next/link";

export const metadata: Metadata = {
  title: "NLP Intelligence — Social Monitoring Analysis",
  description: "Content analysis, topic modeling, and network analysis for social media monitoring",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="mn">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
      </head>
      <body>
        <div className="app-container">
          <nav className="navbar">
            <Link href="/" className="navbar-brand">
              <h1>🧠 NLP Intelligence</h1>
              <span>v1.0</span>
            </Link>
            <div className="navbar-links">
              <Link href="/" className="nav-link" id="nav-dashboard">
                📊 Dashboard
              </Link>
              <Link href="/admin" className="nav-link" id="nav-admin">
                ⚙️ Admin
              </Link>
            </div>
          </nav>
          <main className="main-content">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
