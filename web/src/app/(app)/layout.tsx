import { getMe } from '@/lib/api'
import { redirect } from 'next/navigation'
import Sidebar from '@/components/Sidebar'

export const dynamic = 'force-dynamic'

export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const user = await getMe()
  if (!user) redirect('/')

  return (
    <div className="flex h-screen gap-4 p-4">
      <Sidebar user={user} />
      <main className="flex-1 overflow-auto">{children}</main>
    </div>
  )
}
