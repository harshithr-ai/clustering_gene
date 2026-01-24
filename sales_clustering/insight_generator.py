"""
Insight Generator Module
Generates executive-level insights and actionable recommendations from clustering results
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import config


class InsightGenerator:
    """Generate CEO-level insights from statistical analysis"""
    
    def __init__(self, industry: str = 'default'):
        self.industry = industry
        self.config = config.INDUSTRY_CONFIGS.get(industry, config.INDUSTRY_CONFIGS['default'])
        self.segment_term = self.config['segment_terminology']
        
    def generate_executive_summary(self, clustering_result: Dict, 
                                   statistical_analysis: Dict,
                                   feature_names: List[str]) -> Dict:
        """
        Generate comprehensive executive summary
        
        Args:
            clustering_result: Results from clustering engine
            statistical_analysis: Results from statistical analyzer
            feature_names: List of feature names
            
        Returns:
            Dictionary with executive insights
        """
        summary = {
            'overview': self._create_overview(clustering_result, statistical_analysis),
            'segment_intelligence': self._generate_segment_intelligence(
                clustering_result, statistical_analysis, feature_names
            ),
            'strategic_insights': self._generate_strategic_insights(
                clustering_result, statistical_analysis
            ),
            'actionable_recommendations': self._generate_recommendations(
                clustering_result, statistical_analysis
            ),
            'risk_opportunities': self._identify_risks_and_opportunities(
                clustering_result, statistical_analysis
            ),
            'performance_metrics': self._calculate_performance_metrics(
                clustering_result, statistical_analysis
            )
        }
        
        return summary
    
    def _create_overview(self, clustering_result: Dict, 
                        statistical_analysis: Dict) -> Dict:
        """Create high-level executive overview"""
        metrics = clustering_result['metrics']
        n_segments = clustering_result['n_clusters']
        
        # Quality assessment
        quality_score = self._calculate_quality_score(metrics)
        
        # Business impact assessment
        variance_explained = metrics.get('r_squared', 0) * 100
        
        overview = {
            'executive_summary': self._compose_executive_narrative(
                n_segments, quality_score, variance_explained, statistical_analysis
            ),
            'segmentation_quality': quality_score,
            'confidence_level': self._assess_confidence(metrics),
            'variance_explained_pct': variance_explained,
            'statistical_validity': metrics.get('silhouette_score', 0) > config.SILHOUETTE_FAIR,
            'business_readiness': quality_score >= 0.7
        }
        
        return overview
    
    def _compose_executive_narrative(self, n_segments: int, quality_score: float, 
                                    variance_explained: float, 
                                    statistical_analysis: Dict) -> str:
        """Compose executive narrative"""
        quality_desc = self._describe_quality(quality_score)
        
        narrative = (
            f"Analysis has identified {n_segments} statistically distinct {self.segment_term.lower()}s "
            f"with {quality_desc} separation quality (score: {quality_score:.2f}/1.00). "
            f"These segments explain {variance_explained:.1f}% of the total variance in the data, "
            f"indicating {'strong' if variance_explained > 60 else 'moderate' if variance_explained > 40 else 'modest'} "
            f"structural differentiation. "
        )
        
        # Add balance assessment
        balance = statistical_analysis['segment_sizes'].get('balance_metrics', {})
        if balance:
            balance_score = balance.get('balance_score', 0)
            if balance_score > 0.7:
                narrative += f"Segment distribution is well-balanced (balance index: {balance_score:.2f}), "
                narrative += "suggesting natural clustering without dominant outlier groups. "
            else:
                narrative += f"Segment distribution shows concentration (balance index: {balance_score:.2f}), "
                narrative += "indicating potential dominant patterns in the data. "
        
        # Add statistical validity
        if statistical_analysis.get('statistical_tests', {}).get('global_significance', {}).get('is_significant'):
            narrative += "Statistical tests confirm significant inter-segment differences. "
        
        return narrative
    
    def _describe_quality(self, quality_score: float) -> str:
        """Describe quality score in business terms"""
        if quality_score >= 0.8:
            return "excellent"
        elif quality_score >= 0.6:
            return "good"
        elif quality_score >= 0.4:
            return "fair"
        else:
            return "moderate"
    
    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate overall quality score (0-1)"""
        scores = []
        
        # Silhouette score (weighted 40%)
        silhouette = metrics.get('silhouette_score', 0)
        if not np.isnan(silhouette):
            scores.append(max(0, (silhouette + 1) / 2) * 0.4)  # Normalize from [-1,1] to [0,1]
        
        # R-squared (weighted 30%)
        r_squared = metrics.get('r_squared', 0)
        if not np.isnan(r_squared):
            scores.append(r_squared * 0.3)
        
        # Separation index (weighted 20%)
        sep_index = metrics.get('separation_index', 0)
        if not np.isnan(sep_index):
            scores.append(min(1.0, sep_index / 3) * 0.2)  # Normalize
        
        # Balance score (weighted 10%)
        balance = metrics.get('balance_score', 0.5)
        if not np.isnan(balance):
            scores.append(balance * 0.1)
        
        return sum(scores) if scores else 0.0
    
    def _assess_confidence(self, metrics: Dict) -> str:
        """Assess confidence level in clustering results"""
        silhouette = metrics.get('silhouette_score', 0)
        
        if silhouette >= config.SILHOUETTE_EXCELLENT:
            return "High - Results are highly reliable for decision-making"
        elif silhouette >= config.SILHOUETTE_GOOD:
            return "Moderate - Results provide actionable insights with reasonable confidence"
        elif silhouette >= config.SILHOUETTE_FAIR:
            return "Fair - Results require validation but show meaningful patterns"
        else:
            return "Low - Additional data or features may be needed for robust segmentation"
    
    def _generate_segment_intelligence(self, clustering_result: Dict,
                                       statistical_analysis: Dict,
                                       feature_names: List[str]) -> List[Dict]:
        """Generate detailed intelligence for each segment"""
        profiles = statistical_analysis.get('segment_profiles', {})
        sizes = statistical_analysis.get('segment_sizes', {})
        
        intelligence = []
        
        for segment_name, profile in profiles.items():
            size_info = sizes.get(segment_name, {})
            
            # Generate characterization
            characterization = self._characterize_segment(
                segment_name, profile, size_info, feature_names
            )
            
            # Generate business implications
            implications = self._derive_business_implications(
                segment_name, profile, size_info
            )
            
            intelligence.append({
                'segment': segment_name,
                'size': size_info.get('count', 0),
                'market_share_pct': size_info.get('percentage', 0),
                'characterization': characterization,
                'key_differentiators': profile.get('distinguishing_features', [])[:3],
                'business_implications': implications,
                'homogeneity_index': profile.get('homogeneity', 0),
                'intervention_priority': self._calculate_priority(size_info, profile)
            })
        
        # Sort by market share
        intelligence.sort(key=lambda x: x['market_share_pct'], reverse=True)
        
        return intelligence
    
    def _characterize_segment(self, segment_name: str, profile: Dict, 
                             size_info: Dict, feature_names: List[str]) -> str:
        """Create narrative characterization of segment"""
        size = size_info.get('count', 0)
        pct = size_info.get('percentage', 0)
        
        # Get top distinguishing features
        features = profile.get('distinguishing_features', [])[:3]
        
        if not features:
            return f"{segment_name} comprises {size} records ({pct:.1f}% of population)."
        
        char = f"{segment_name} represents {pct:.1f}% of the population ({size} records) and is "
        
        # Build feature description
        feature_desc = []
        for feat in features:
            direction = feat['direction'].lower()
            impact = feat['impact_level'].lower()
            feature_name = feat['feature']
            effect = abs(feat['effect_size'])
            
            if effect > 0.8:
                magnitude = "significantly"
            elif effect > 0.5:
                magnitude = "notably"
            else:
                magnitude = "moderately"
            
            feature_desc.append(f"{magnitude} {direction} in {feature_name}")
        
        char += ", ".join(feature_desc[:2])
        
        # Add homogeneity assessment
        homogeneity = profile.get('homogeneity', 0)
        if homogeneity > 0.8:
            char += ". This segment exhibits high internal consistency."
        elif homogeneity < 0.5:
            char += ". This segment shows notable internal variation."
        
        return char
    
    def _derive_business_implications(self, segment_name: str, 
                                     profile: Dict, size_info: Dict) -> List[str]:
        """Derive business implications from segment characteristics"""
        implications = []
        
        size_pct = size_info.get('percentage', 0)
        is_actionable = size_info.get('is_actionable', False)
        homogeneity = profile.get('homogeneity', 0)
        
        # Size-based implications
        if size_pct > 30:
            implications.append(
                f"Dominant segment representing {size_pct:.1f}% of population - "
                "Primary focus area for strategy development"
            )
        elif size_pct > 15:
            implications.append(
                f"Significant segment ({size_pct:.1f}%) - "
                "Warrants dedicated resource allocation"
            )
        elif is_actionable:
            implications.append(
                f"Niche segment ({size_pct:.1f}%) - "
                "Potential for specialized targeting"
            )
        else:
            implications.append(
                f"Minor segment ({size_pct:.1f}%) - "
                "Monitor for growth trends"
            )
        
        # Homogeneity-based implications
        if homogeneity > 0.8:
            implications.append(
                "High homogeneity enables precise targeting and standardized approaches"
            )
        elif homogeneity < 0.5:
            implications.append(
                "Internal diversity suggests need for sub-segmentation or flexible strategies"
            )
        
        # Feature-based implications
        features = profile.get('distinguishing_features', [])
        if features:
            top_feature = features[0]
            if top_feature['effect_size'] > 0.8:
                implications.append(
                    f"Strong differentiation on {top_feature['feature']} "
                    f"provides clear intervention lever"
                )
        
        return implications
    
    def _calculate_priority(self, size_info: Dict, profile: Dict) -> int:
        """Calculate intervention priority (1-5, 5 is highest)"""
        priority = 0
        
        # Size contribution (0-2 points)
        size_pct = size_info.get('percentage', 0)
        if size_pct > 25:
            priority += 2
        elif size_pct > 10:
            priority += 1
        
        # Homogeneity (0-2 points)
        homogeneity = profile.get('homogeneity', 0)
        if homogeneity > 0.8:
            priority += 2
        elif homogeneity > 0.6:
            priority += 1
        
        # Actionability (0-1 point)
        if size_info.get('is_actionable', False):
            priority += 1
        
        return min(5, priority)
    
    def _generate_strategic_insights(self, clustering_result: Dict,
                                    statistical_analysis: Dict) -> List[str]:
        """Generate strategic insights for executive decision-making"""
        insights = []
        
        metrics = clustering_result['metrics']
        n_segments = clustering_result['n_clusters']
        
        # Segmentation structure insight
        if n_segments <= 3:
            insights.append(
                f"Market exhibits {n_segments} primary segments, suggesting "
                "clear structural division suitable for macro-level strategy"
            )
        elif n_segments <= 5:
            insights.append(
                f"Market demonstrates {n_segments} distinct segments, indicating "
                "moderate complexity requiring multi-faceted approach"
            )
        else:
            insights.append(
                f"Market shows high fragmentation ({n_segments} segments), "
                "suggesting need for either niche strategies or segment consolidation"
            )
        
        # Separation quality insight
        sep_metrics = statistical_analysis.get('segment_separation', {})
        sep_score = sep_metrics.get('separation_score', 0)
        
        if sep_score > 2.0:
            insights.append(
                "Segments are highly distinct with minimal overlap - "
                "enabling precise targeted interventions with low cross-contamination risk"
            )
        elif sep_score > 1.0:
            insights.append(
                "Moderate segment overlap suggests potential for cross-segment spillover effects - "
                "consider integrated strategies"
            )
        else:
            insights.append(
                "Significant segment overlap indicates fuzzy boundaries - "
                "interventions may benefit multiple segments simultaneously"
            )
        
        # Stability insight
        stability = statistical_analysis.get('segment_stability', {})
        stability_score = stability.get('stability_score', 0)
        
        if stability_score > 0.8:
            insights.append(
                "High segment stability indicates robust structure - "
                "suitable for long-term strategic planning"
            )
        elif stability_score > 0.6:
            insights.append(
                "Moderate segment stability suggests structure is reasonably persistent - "
                "periodic validation recommended"
            )
        else:
            insights.append(
                "Low segment stability indicates dynamic structure - "
                "requires continuous monitoring and adaptive strategies"
            )
        
        # Feature importance insight
        feature_importance = statistical_analysis.get('feature_importance', {})
        if feature_importance.get('anova_ranking'):
            top_features = feature_importance['anova_ranking'][:3]
            sig_features = [f['feature'] for f in top_features if f['is_significant']]
            
            if sig_features:
                insights.append(
                    f"Key differentiating factors are: {', '.join(sig_features)} - "
                    "these represent primary intervention levers"
                )
        
        return insights
    
    def _generate_recommendations(self, clustering_result: Dict,
                                 statistical_analysis: Dict) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Identify high-priority segments
        actionable = statistical_analysis.get('actionable_segments', [])
        
        for segment in actionable:
            segment_name = segment['segment']
            size_pct = segment['size_pct']
            features = segment['high_impact_features']
            
            if features:
                top_feature = features[0]
                
                recommendation = {
                    'target': segment_name,
                    'priority': 'High' if size_pct > 20 else 'Medium',
                    'action': self._formulate_action(segment_name, top_feature, size_pct),
                    'expected_impact': self._estimate_impact(segment, statistical_analysis),
                    'implementation_complexity': self._assess_complexity(segment)
                }
                
                recommendations.append(recommendation)
        
        # Sort by priority and impact
        priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
        recommendations.sort(key=lambda x: (
            priority_order.get(x['priority'], 3),
            -x['expected_impact']
        ))
        
        return recommendations
    
    def _formulate_action(self, segment_name: str, top_feature: Dict, size_pct: float) -> str:
        """Formulate specific action recommendation"""
        feature = top_feature['feature']
        deviation = top_feature['deviation_pct']
        direction = "higher" if deviation > 0 else "lower"
        
        if abs(deviation) > 50:
            intensity = "significantly"
        elif abs(deviation) > 25:
            intensity = "notably"
        else:
            intensity = "moderately"
        
        action = (
            f"Develop targeted intervention for {segment_name} ({size_pct:.1f}% of population) "
            f"focusing on {feature}, which is {intensity} {direction} than baseline. "
        )
        
        if size_pct > 20:
            action += "High market share justifies dedicated resource allocation."
        
        return action
    
    def _estimate_impact(self, segment: Dict, statistical_analysis: Dict) -> float:
        """Estimate potential business impact (0-1 scale)"""
        size_pct = segment['size_pct']
        features = segment['high_impact_features']
        
        # Impact based on segment size (0-0.5)
        size_impact = min(0.5, size_pct / 100)
        
        # Impact based on feature differentiation (0-0.5)
        if features:
            avg_deviation = np.mean([abs(f['deviation_pct']) for f in features])
            feature_impact = min(0.5, avg_deviation / 100)
        else:
            feature_impact = 0
        
        return size_impact + feature_impact
    
    def _assess_complexity(self, segment: Dict) -> str:
        """Assess implementation complexity"""
        n_features = len(segment['high_impact_features'])
        
        if n_features == 1:
            return "Low - Single primary lever"
        elif n_features == 2:
            return "Moderate - Dual-factor approach required"
        else:
            return "High - Multi-faceted intervention needed"
    
    def _identify_risks_and_opportunities(self, clustering_result: Dict,
                                         statistical_analysis: Dict) -> Dict:
        """Identify risks and opportunities"""
        sizes = statistical_analysis.get('segment_sizes', {})
        profiles = statistical_analysis.get('segment_profiles', {})
        
        risks = []
        opportunities = []
        
        # Concentration risk
        for segment_name, size_info in sizes.items():
            if segment_name == 'balance_metrics':
                continue
                
            pct = size_info.get('percentage', 0)
            
            if pct > 60:
                risks.append({
                    'type': 'Concentration Risk',
                    'description': f"{segment_name} dominates with {pct:.1f}% share - "
                                  "over-reliance on single segment creates vulnerability",
                    'severity': 'High'
                })
            
            if pct < 5 and size_info.get('is_actionable'):
                opportunities.append({
                    'type': 'Growth Potential',
                    'description': f"{segment_name} is currently small ({pct:.1f}%) "
                                  "but actionable - potential for expansion",
                    'potential': 'Medium'
                })
        
        # Outlier opportunity
        outlier_segments = [
            (name, profile) for name, profile in profiles.items()
            if profile.get('outlier_ratio', 0) > 0.1
        ]
        
        for segment_name, profile in outlier_segments:
            opportunities.append({
                'type': 'Outlier Investigation',
                'description': f"{segment_name} contains {profile['outlier_ratio']*100:.1f}% outliers - "
                              "potential for sub-segmentation or anomaly investigation",
                'potential': 'High'
            })
        
        return {
            'risks': risks,
            'opportunities': opportunities
        }
    
    def _calculate_performance_metrics(self, clustering_result: Dict,
                                      statistical_analysis: Dict) -> Dict:
        """Calculate key performance indicators"""
        metrics = clustering_result['metrics']
        
        # Calculate composite scores
        kpis = {
            'segmentation_quality_index': self._calculate_quality_score(metrics),
            'variance_explained_pct': metrics.get('r_squared', 0) * 100,
            'segment_separation_score': metrics.get('separation_index', 0),
            'statistical_significance': 'Confirmed' if metrics.get('silhouette_score', 0) > 0.25 else 'Weak',
            'stability_index': statistical_analysis.get('segment_stability', {}).get('stability_score', 0),
            'actionable_segments_count': len(statistical_analysis.get('actionable_segments', [])),
            'business_readiness': 'Production Ready' if self._calculate_quality_score(metrics) >= 0.7 else 'Requires Refinement'
        }
        
        return kpis
