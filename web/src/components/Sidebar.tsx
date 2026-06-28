'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import clsx from 'clsx'

const nav = [
  { href: '/library', label: 'Library', icon: '▤' },
  { href: '/ask', label: 'Ask', icon: '◎' },
  { href: '/labels', label: 'Labels', icon: '#' },
  { href: '/features', label: 'Features', icon: '+' },
]

export default function Sidebar({ user }: { user: { name: string; email: string } }) {
  const pathname = usePathname()
  return (
    <aside className="glass h-full w-56 flex flex-col p-4 gap-2 shrink-0">
      <div className="mb-4 px-2">
        <span className="text-lg font-bold tracking-tight">VideoVault</span>
      </div>
      {nav.map(({ href, label, icon }) => (
        <Link
          key={href}
          href={href}
          className={clsx(
            'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors',
            pathname.startsWith(href)
              ? 'bg-white/14 text-white font-medium'
              : 'text-white/55 hover:text-white hover:bg-white/8'
          )}
        >
          <span>{icon}</span>
          {label}
        </Link>
      ))}
      <div className="mt-auto glass-sm truncate px-3 py-2 text-xs text-white/50">
        {user.email}
      </div>
    </aside>
  )
}
