import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpStatus, value
from typing import List, Dict, Tuple
import time
from dataclasses import dataclass

# Constants
PRIORITY_WEIGHTS = {'high': 3, 'medium': 2, 'low': 1}
CSV_CONFIG = {
    'employees': {
        'path': "employees.csv",
        'dtype': {'Skills': str}
    },
    'tickets': {
        'path': "tickets.csv",
        'dtype': {'Response Time': int}
    }
}

@dataclass
class AssignmentResult:
    assignment_df: pd.DataFrame
    optimization_time: float
    status: str
    model: LpProblem

class TicketAssignmentOptimizer:
    def __init__(self):
        self.employees_df = None
        self.tickets_df = None
        self.assignments = None
        self.model = None
    
    def load_data(self) -> None:
        """Load and preprocess employee and ticket data"""
        self.employees_df = pd.read_csv(
            CSV_CONFIG['employees']['path'],
            dtype=CSV_CONFIG['employees']['dtype']
        )
        self.tickets_df = pd.read_csv(
            CSV_CONFIG['tickets']['path'],
            dtype=CSV_CONFIG['tickets']['dtype']
        )
        
        # Preprocess data
        self.employees_df['Skills'] = self.employees_df['Skills'].str.split(',')
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Validate input data for required columns and values"""
        required_employee_cols = ['Employee', 'Skills']
        required_ticket_cols = ['Ticket ID', 'Issue Type', 'Priority', 'Response Time']
        
        for col in required_employee_cols:
            if col not in self.employees_df.columns:
                raise ValueError(f"Missing required column in employees data: {col}")
                
        for col in required_ticket_cols:
            if col not in self.tickets_df.columns:
                raise ValueError(f"Missing required column in tickets data: {col}")
                
        if len(self.employees_df) == 0:
            raise ValueError("Employee data is empty")
            
        if len(self.tickets_df) == 0:
            raise ValueError("Ticket data is empty")
    
    def _generate_valid_assignments(self) -> List[Tuple[int, int]]:
        """Generate all possible valid ticket-employee assignments based on skills"""
        assignments = []
        ticket_indices = self.tickets_df.index.tolist()
        employee_indices = self.employees_df.index.tolist()
        
        for t_idx in ticket_indices:
            ticket_issue_type = self.tickets_df.loc[t_idx, 'Issue Type']
            for e_idx in employee_indices:
                if ticket_issue_type in self.employees_df.loc[e_idx, 'Skills']:
                    assignments.append((t_idx, e_idx))
        
        if not assignments:
            raise ValueError("No valid assignments found - skill mismatch between tickets and employees")
            
        return assignments
    
    def optimize_assignments(self) -> AssignmentResult:
        """Run the optimization process"""
        start_time = time.time()
        
        # Initialize model
        self.model = LpProblem("Ticket_Assignment_Optimization", LpMinimize)
        self.assignments = self._generate_valid_assignments()
        
        # Decision variables
        x = LpVariable.dicts("assign", self.assignments, cat=LpBinary)
        
        # Objective function
        self._set_objective_function(x)
        
        # Constraints
        self._add_constraints(x)
        
        # Solve the model
        self.model.solve()
        status = LpStatus[self.model.status]
        
        # Process results
        assignment_df = self._process_results(x) if status == 'Optimal' else pd.DataFrame()
        
        return AssignmentResult(
            assignment_df=assignment_df,
            optimization_time=time.time() - start_time,
            status=status,
            model=self.model
        )
    
    def _set_objective_function(self, x: Dict[Tuple[int, int], LpVariable]) -> None:
        """Set the minimization objective function"""
        objective_terms = []
        
        for (t_idx, e_idx) in self.assignments:
            response_time = self.tickets_df.loc[t_idx, 'Response Time']
            priority = self.tickets_df.loc[t_idx, 'Priority']
            weight = PRIORITY_WEIGHTS.get(priority, 1)  # Default to 1 if priority not found
            objective_terms.append(x[(t_idx, e_idx)] * response_time * weight)
        
        self.model += lpSum(objective_terms)
    
    def _add_constraints(self, x: Dict[Tuple[int, int], LpVariable]) -> None:
        """Add all constraints to the model"""
        self._add_ticket_assignment_constraints(x)
        self._add_workload_balance_constraints(x)
    
    def _add_ticket_assignment_constraints(self, x: Dict[Tuple[int, int], LpVariable]) -> None:
        """Ensure each ticket is assigned to exactly one employee"""
        ticket_indices = self.tickets_df.index.unique()
        
        for t_idx in ticket_indices:
            relevant_assignments = [a for a in self.assignments if a[0] == t_idx]
            if not relevant_assignments:
                raise ValueError(f"No valid employees for ticket index {t_idx}")
            self.model += lpSum(x[a] for a in relevant_assignments) == 1
    
    def _add_workload_balance_constraints(self, x: Dict[Tuple[int, int], LpVariable]) -> None:
        """Ensure fair workload distribution among employees"""
        avg_load = len(self.tickets_df) / len(self.employees_df)
        max_load = int(avg_load + 1)
        min_load = max(1, int(avg_load - 0.5))  # At least 1, or floor of average
        
        employee_indices = self.employees_df.index.unique()
        
        for e_idx in employee_indices:
            relevant_assignments = [a for a in self.assignments if a[1] == e_idx]
            if not relevant_assignments:
                continue  # Employee has no valid assignments (shouldn't happen due to skill check)
            
            self.model += lpSum(x[a] for a in relevant_assignments) <= max_load
            self.model += lpSum(x[a] for a in relevant_assignments) >= min_load
    
    def _process_results(self, x: Dict[Tuple[int, int], LpVariable]) -> pd.DataFrame:
        """Convert optimization results to a DataFrame"""
        results = []
        
        for (t_idx, e_idx), var in x.items():
            if var.varValue == 1:
                ticket = self.tickets_df.loc[t_idx]
                employee = self.employees_df.loc[e_idx]
                
                results.append({
                    'Ticket ID': ticket['Ticket ID'],
                    'Issue Type': ticket['Issue Type'],
                    'Assigned To': employee['Employee'],
                    'Response Time': ticket['Response Time'],
                    'Priority': ticket['Priority'],
                    'Priority Weight': PRIORITY_WEIGHTS[ticket['Priority']]
                })
        
        return pd.DataFrame(results)
    
    def visualize_workload(self, assignment_df: pd.DataFrame) -> None:
        """Generate workload distribution visualization"""
        if assignment_df.empty:
            print("No assignments to visualize")
            return
        
        plt.figure(figsize=(10, 6))
        workload = assignment_df.groupby('Assigned To').size().reset_index(name='Tickets Assigned')
        
        ax = sns.barplot(
            data=workload,
            x='Assigned To',
            y='Tickets Assigned',
            palette='viridis'
        )
        
        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0, 10),
                textcoords='offset points'
            )
        
        plt.title("Optimized Workload Distribution")
        plt.ylabel("Number of Tickets")
        plt.xlabel("Employee")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def compare_with_random_assignment(self, optimized_df: pd.DataFrame) -> None:
        """Compare optimized assignment with random assignment"""
        if optimized_df.empty:
            print("No optimized assignments to compare")
            return
        
        # Create random assignment for comparison
        random_results = []
        for _, ticket in self.tickets_df.iterrows():
            valid_employees = self.employees_df[
                self.employees_df['Skills'].apply(lambda x: ticket['Issue Type'] in x)
            ]
            if not valid_employees.empty:
                random_emp = valid_employees.sample(1).iloc[0]
                random_results.append({
                    'Ticket ID': ticket['Ticket ID'],
                    'Response Time': ticket['Response Time'],
                    'Priority Weight': PRIORITY_WEIGHTS[ticket['Priority']]
                })
        
        random_df = pd.DataFrame(random_results)
        
        # Calculate metrics
        comparison = pd.DataFrame({
            'Method': ['Optimized', 'Random'],
            'Total Weighted Time': [
                (optimized_df['Response Time'] * optimized_df['Priority Weight']).sum(),
                (random_df['Response Time'] * random_df['Priority Weight']).sum()
            ],
            'Max Single Workload': [
                optimized_df.groupby('Assigned To').size().max(),
                len(random_df)  # Random doesn't track assignment
            ]
        })
        
        # Visualization
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.barplot(data=comparison, x='Method', y='Total Weighted Time')
        plt.title("Total Priority-Weighted Response Time")
        
        plt.subplot(1, 2, 2)
        sns.barplot(data=comparison, x='Method', y='Max Single Workload')
        plt.title("Maximum Tickets Assigned to One Employee")
        
        plt.tight_layout()
        plt.show()


def main():
    """Main execution function"""
    try:
        # Initialize and run optimizer
        optimizer = TicketAssignmentOptimizer()
        optimizer.load_data()
        
        print("Employees Preview:")
        print(optimizer.employees_df.head())
        print("\nTickets Preview:")
        print(optimizer.tickets_df.head())
        
        # Run optimization
        result = optimizer.optimize_assignments()
        
        print(f"\nOptimization Status: {result.status}")
        print(f"Optimization Time: {result.optimization_time:.2f} seconds")
        
        if not result.assignment_df.empty:
            print("\nOptimized Assignments:")
            print(result.assignment_df)
            
            # Visualizations
            optimizer.visualize_workload(result.assignment_df)
            optimizer.compare_with_random_assignment(result.assignment_df)
        else:
            print("No optimal solution found")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()