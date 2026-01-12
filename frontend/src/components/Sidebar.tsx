import { NavLink } from "react-router-dom";
import { LayoutDashboard, Calculator, Building2, Pill, TrendingUp } from "lucide-react";

const Sidebar = () => {
  const navItems = [
    { to: "/", icon: LayoutDashboard, label: "Dashboard" },
    { to: "/distribute", icon: Calculator, label: "Calculate Distribution" },
    { to: "/hospitals", icon: Building2, label: "Hospitals" },
    { to: "/metrics", icon: TrendingUp, label: "Model Metrics" },
  ];

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 bg-sidebar border-r border-sidebar-border flex flex-col">
      <div className="p-6 border-b border-sidebar-border">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center">
            <Pill className="w-5 h-5 text-primary-foreground" />
          </div>
          <div>
            <h1 className="font-bold text-sidebar-foreground">PharmaDist</h1>
            <p className="text-xs text-muted-foreground">Distribution System</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 p-4 space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                isActive
                  ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                  : "text-sidebar-foreground hover:bg-sidebar-accent/50"
              }`
            }
          >
            <item.icon className="w-5 h-5" />
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="p-4 border-t border-sidebar-border">
        <div className="px-4 py-3 rounded-lg bg-sidebar-accent/30">
          <p className="text-xs text-muted-foreground">Local Development</p>
          <p className="text-sm font-medium text-sidebar-foreground">v1.0.0</p>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
