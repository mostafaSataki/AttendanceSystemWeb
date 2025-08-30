import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Face Recognition - Attendance System",
  description: "Modern face recognition attendance system built with Next.js, FastAPI, and advanced computer vision. Track attendance automatically.",
  keywords: ["Face Recognition", "Next.js", "TypeScript", "Tailwind CSS", "shadcn/ui", "Attendance System", "React", "Computer Vision"],
  authors: [{ name: "Face Recognition Team" }],
  openGraph: {
    title: "Face Recognition",
    description: "Automated attendance tracking with face recognition",
    url: "https://localhost:3000",
    siteName: "Face Recognition",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Face Recognition",
    description: "Automated attendance tracking with face recognition",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
