import { getMe } from '@/lib/api'
import { redirect } from 'next/navigation'

export const dynamic = 'force-dynamic'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'

export default async function Home() {
  const user = await getMe()
  if (user) redirect('/library')

  return (
    <main className="min-h-screen flex items-center justify-center">
      <div className="glass p-10 max-w-md w-full text-center space-y-6">
        <h1 className="text-3xl font-bold tracking-tight">VideoVault</h1>
        <p className="text-white/60 text-sm">
          Query your videos with natural language. Find the exact frame you need.
        </p>
        <a
          href={`${API_URL}/oauth2/authorization/google`}
          className="block w-full py-3 px-6 rounded-xl bg-white text-gray-900 font-semibold
                     hover:bg-white/90 transition-colors"
        >
          Sign in with Google
        </a>
      </div>
    </main>
  )
}
